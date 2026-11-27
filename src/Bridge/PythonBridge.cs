using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace Bridge;

public sealed class PythonBridge : IDisposable
{
    private readonly string _pythonExe;
    private readonly string _pythonScript;
    private readonly string _configPath;
    private readonly string _bridgeId = Guid.NewGuid().ToString("N");
    private readonly ReaderWriterLockSlim _lifecycleLock = new();
    private readonly SemaphoreSlim _inflight;
    private JsonLineTransport? _transport;
    private volatile bool _closing;
    private volatile bool _disposed;

    public int MaxConcurrentRequests { get; }
    public int RequestTimeoutMilliseconds { get; set; } = 120000;

    private Process? _process;
    private StreamWriter? _stdin;
    private StreamReader? _stdout;
    private bool _initialized;

    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = null,
        WriteIndented = false
    };

    // Python桥接器
    public PythonBridge(string pythonExe, string pythonScript) : this(pythonExe, pythonScript, 13) { }

    public PythonBridge(string pythonExe, string pythonScript, int maxConcurrentRequests)
    {
        if (maxConcurrentRequests < 2 || maxConcurrentRequests > 13)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrentRequests), "Use 2 through 13 concurrent requests.");
        MaxConcurrentRequests = maxConcurrentRequests;
        _inflight = new SemaphoreSlim(maxConcurrentRequests, maxConcurrentRequests);
        _pythonExe = pythonExe;
        _pythonScript = pythonScript;
        _configPath = ResolveConfigPath(pythonScript);
        // 创建子接口
        TwoD = new Bridge2D(this);
        ThreeD = new Bridge3D(this);
    }

    public Bridge2D TwoD { get; }

    public Bridge3D ThreeD { get; }

    public DetectResponse ReleaseBoard(string boardId)
    {
        if (string.IsNullOrWhiteSpace(boardId)) throw new ArgumentException("Board ID is empty.", nameof(boardId));
        using (EnterDetection())
        {
            var watch = Stopwatch.StartNew();
            return SendDetectLocked(new ReleaseBoardRequest { BoardId = boardId }, watch, 0);
        }
    }

    // 初始化通信服务
    public BridgeBaseResponse Init()
    {
        using (EnterLifecycleOperation())
        {
            if (_disposed) throw new ObjectDisposedException(nameof(PythonBridge));
            _closing = false;
            Start();
            InitRequest req = new();
            BridgeBaseResponse response = Send<InitRequest, BridgeBaseResponse>(
                req,
                out double roundtripMs);

            response.CsharpRoundtripMs = roundtripMs;

            if (!response.Ok)
            {
                throw new Exception("Python init failed: " + response.Error);
            }

            _initialized = true;
            return response;
        }
    }

    // 初始化算法服务
    public BridgeBaseResponse InitAlgorithm()
    {
        return InitAlgorithm("board");
    }

    public BridgeBaseResponse InitAlgorithm(string stage)
    {
        return InitAlgorithm(stage, _configPath);
    }

    /// <summary>Initialize the detector using the specified product configuration manifest.</summary>
    public BridgeBaseResponse InitAlgorithm(string stage, string configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath))
            throw new ArgumentException("Product configuration path is empty.", nameof(configPath));

        InitAlgorithmRequest req = CreateInitAlgorithmRequest(stage, Path.GetFullPath(configPath));
        if (!_initialized || _process == null || _process.HasExited)
        {
            Init();
        }

        using (EnterLifecycleOperation())
        {
            BridgeBaseResponse response = Send<InitAlgorithmRequest, BridgeBaseResponse>(
                req,
                out double roundtripMs);

            response.CsharpRoundtripMs = roundtripMs;

            if (!response.Ok)
            {
                throw new Exception($"Python {stage} algorithm init failed: " + response.Error);
            }

            return response;
        }
    }

    // 释放资源
    public BridgeBaseResponse Release()
    {
        _closing = true;
        using var lifecycle = EnterLifecycleOperation();
        if (_process == null)
        {
            return new BridgeBaseResponse
            {
                Ok = true,
                Cmd = "release",
                Message = "python process is not started"
            };
        }

        if (_process.HasExited)
        {
            CleanupProcessObjects();

            return new BridgeBaseResponse
            {
                Ok = true,
                Cmd = "release",
                Message = "python process already exited"
            };
        }

        BridgeBaseResponse? response = null;

        {
            try
            {
                response = Send<ReleaseRequest, BridgeBaseResponse>(
                    new ReleaseRequest(),
                    out double roundtripMs);
                response.CsharpRoundtripMs = roundtripMs;
            }
            catch
            {
                // 即使Python已经异常 也继续走下面的进程清理流程
            }

            try
            {
                if (!_process.WaitForExit(2000))
                {
                    _process.Kill();
                    _process.WaitForExit(1000);
                }
            }
            finally
            {
                CleanupProcessObjects();
            }
        }

        return response ?? new BridgeBaseResponse
        {
            Ok = true,
            Cmd = "release",
            Message = "python process closed"
        };
    }

    // 重新加载缺陷配置 更新配置
    public BridgeBaseResponse ReloadDefectConfig(string configPath)
    {
        if (string.IsNullOrWhiteSpace(configPath))
        {
            throw new ArgumentException("Config path is empty.", nameof(configPath));
        }

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException("Config file not found.", configPath);
        }

        return ReloadDefectConfigText(
            File.ReadAllText(configPath, Encoding.UTF8),
            configPath);
    }

    // 重新加载配置内容
    public BridgeBaseResponse ReloadDefectConfigText(
        string configText,
        string? configPath = null)
    {
        EnsureInitialized();

        if (string.IsNullOrWhiteSpace(configText))
        {
            throw new ArgumentException("Config text is empty.", nameof(configText));
        }

        var req = new ReloadDefectConfigRequest
        {
            ConfigPath = configPath,
            ConfigText = configText
        };

        using (EnterLifecycleOperation())
        {
            BridgeBaseResponse response = Send<ReloadDefectConfigRequest, BridgeBaseResponse>(
                req,
                out double roundtripMs);
            response.CsharpRoundtripMs = roundtripMs;

            if (!response.Ok)
            {
                throw new Exception("Python reload defect config failed: " + response.Error);
            }

            return response;
        }
    }

    // 2D 检测模块
    // 校验imageBytes 写入Bridge自己管理的共享内存
    internal DetectResponse Detect2D(
        byte[] imageBytes,
        Detect2DRequest request,
        string bufferKey,
        double imageReadMs = 0)
    {
        EnsureInitialized();

        if (imageBytes.Length == 0)
        {
            throw new ArgumentException("Image bytes is empty.", nameof(imageBytes));
        }

        using (EnterDetection())
        {
            var totalWatch = Stopwatch.StartNew();
            var shmWriteWatch = Stopwatch.StartNew();
            SharedMemoryBuffer buffer = CreateSharedMemoryBuffer(bufferKey, imageBytes.Length);
            try
            {
            buffer.Accessor.WriteArray(0, imageBytes, 0, imageBytes.Length);
            shmWriteWatch.Stop();

            request.ShmName = buffer.Name;
            request.ImageSize = imageBytes.Length;

            return SendDetectLocked(
                request,
                totalWatch,
                shmWriteWatch.Elapsed.TotalMilliseconds,
                imageReadMs,
                buffer);
            }
            catch { buffer.Dispose(); throw; }
        }
    }

    // 如果外部已经写好共享内存只发送元数据
    internal DetectResponse Detect2DSharedMemory(Detect2DRequest request)
    {
        EnsureInitialized();

        using (EnterDetection())
        {
            var totalWatch = Stopwatch.StartNew();
            return SendDetectLocked(request, totalWatch, sharedMemoryWriteMs: 0);
        }
    }

    internal DetectResponse Detect2DFile(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image file path is empty.", nameof(imagePath));
        }

        var imageReadWatch = Stopwatch.StartNew();
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        imageReadWatch.Stop();
        var request = new Detect2DRequest
        {
            ImageFormat = "file"
        };

        return Detect2D(imageBytes, request, "2d_file", imageReadWatch.Elapsed.TotalMilliseconds);
    }

    // Streaming 2D entry: one board/piece at a time. The identifiers are
    // carried in the protocol so Python can cache the processed mask by board.
    internal DetectResponse Detect2DPieceFile(
        string boardId,
        int pieceId,
        string imagePath,
        object? threshold = null)
    {
        if (string.IsNullOrWhiteSpace(boardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(boardId));
        }
        if (pieceId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pieceId));
        }
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image file path is empty.", nameof(imagePath));
        }

        var imageReadWatch = Stopwatch.StartNew();
        byte[] imageBytes = File.ReadAllBytes(imagePath);
        imageReadWatch.Stop();
        return Detect2DPiece(
            imageBytes,
            new Detect2DRequest
            {
                Cmd = "detect_2d_piece",
                BoardId = boardId,
                PieceId = pieceId,
                ImageFormat = "file",
                Threshold = threshold
            },
            $"stream_2d_{boardId}_{pieceId}",
            imageReadWatch.Elapsed.TotalMilliseconds);
    }

    internal DetectResponse Detect2DPiece(
        byte[] imageBytes,
        Detect2DRequest request,
        string bufferKey,
        double imageReadMs = 0)
    {
        if (string.IsNullOrWhiteSpace(request.BoardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(request));
        }
        if (request.PieceId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request.PieceId));
        }
        return Detect2D(imageBytes, request, bufferKey, imageReadMs);
    }

    // 3D 检测模块
    internal DetectResponse Detect3DValues(ushort[] values, Detect3DRequest request, int width, int height)
    {
        int size = PackedDepthInput.ByteCount(width, height);
        if (values == null || values.LongLength * sizeof(ushort) != size)
            throw new ArgumentException("3D array length must equal width * height.", nameof(values));
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("3D ushort transport requires little-endian storage.");
        EnsureInitialized();
        using (EnterDetection())
        {
            var totalWatch = Stopwatch.StartNew();
            var writeWatch = Stopwatch.StartNew();
            var buffer = CreateSharedMemoryBuffer("3d_uint16", size);
            try
            {
                buffer.Accessor.WriteArray(0, values, 0, values.Length);
                writeWatch.Stop();
                request.ShmName = buffer.Name;
                request.DataSize = size;
                request.DataFormat = "uint16";
                request.ImageWidth = width;
                request.ImageHeight = height;
                return SendDetectLocked(request, totalWatch, writeWatch.Elapsed.TotalMilliseconds,
                    requestLifetime: buffer);
            }
            catch { buffer.Dispose(); throw; }
        }
    }

    internal DetectResponse DetectStreamed3DFile(string dataPath, Detect3DRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.BoardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(request));
        }
        if (string.IsNullOrWhiteSpace(dataPath))
        {
            throw new ArgumentException("3D data file path is empty.", nameof(dataPath));
        }
        if (!File.Exists(dataPath))
        {
            throw new FileNotFoundException("3D data file not found.", dataPath);
        }

        var imageReadWatch = Stopwatch.StartNew();
        byte[] dataBytes = PackedDepthInput.ReadFile(dataPath, out int width, out int height);
        imageReadWatch.Stop();
        request.ImageWidth = width;
        request.ImageHeight = height;
        return DetectStreamed3D(dataBytes, request, $"stream_3d_{request.BoardId}", imageReadWatch.Elapsed.TotalMilliseconds);
    }

    private DetectResponse DetectStreamed3D(
        byte[] dataBytes,
        Detect3DRequest request,
        string bufferKey,
        double imageReadMs = 0)
    {
        EnsureInitialized();
        if (string.IsNullOrWhiteSpace(request.BoardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(request));
        }
        if (dataBytes == null || dataBytes.Length == 0)
        {
            throw new ArgumentException("3D data bytes is empty.", nameof(dataBytes));
        }
        PackedDepthInput.Validate(dataBytes.Length, request.ImageWidth, request.ImageHeight, request.DataFormat);

        using (EnterDetection())
        {
            var totalWatch = Stopwatch.StartNew();
            var shmWriteWatch = Stopwatch.StartNew();
            SharedMemoryBuffer buffer = CreateSharedMemoryBuffer(bufferKey, dataBytes.Length);
            try
            {
            buffer.Accessor.WriteArray(0, dataBytes, 0, dataBytes.Length);
            shmWriteWatch.Stop();

            request.ShmName = buffer.Name;
            request.DataSize = dataBytes.Length;
            DetectResponse response = SendDetectLocked(
                request,
                totalWatch,
                shmWriteWatch.Elapsed.TotalMilliseconds,
                imageReadMs,
                buffer);
            return response;
            }
            catch { buffer.Dispose(); throw; }
        }
    }

    internal DetectResponse DetectStreamed3DSharedMemory(Detect3DRequest request)
    {
        EnsureInitialized();
        if (string.IsNullOrWhiteSpace(request.BoardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(request));
        }
        Validate3DSharedMemory(request);

        using (EnterDetection())
        {
            var totalWatch = Stopwatch.StartNew();
            DetectResponse response = SendDetectLocked(
                request,
                totalWatch,
                sharedMemoryWriteMs: 0);
            // The caller owns the external mapping.
            return response;
        }
    }

    private static void Validate3DSharedMemory(Detect3DRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.ShmName))
            throw new ArgumentException("3D shared-memory name is empty.");
        PackedDepthInput.Validate(request.DataSize, request.ImageWidth, request.ImageHeight, request.DataFormat);
    }

    private DetectResponse SendDetectLocked<TRequest>(
        TRequest request,
        Stopwatch totalWatch,
        double sharedMemoryWriteMs,
        double imageReadMs = 0,
        IDisposable? requestLifetime = null)
    {
        // Request-scoped memory remains alive until its matching response arrives.
        DetectResponse response = Send<TRequest, DetectResponse>(
            request,
            out double roundtripMs,
            requestLifetime ?? HoldRequestMappings(request));
        totalWatch.Stop();

        response.CsharpImageReadMs = imageReadMs;
        response.CsharpSharedMemoryWriteMs = sharedMemoryWriteMs;
        response.CsharpRoundtripMs = roundtripMs;
        response.CsharpTotalMs = totalWatch.Elapsed.TotalMilliseconds;
        AddCsharpTimings(response);

        return response;
    }

    private void Start()
    {
        if (_process != null && !_process.HasExited)
        {
            return;
        }

        if (IsExplicitPath(_pythonExe) && !File.Exists(_pythonExe))
        {
            throw new FileNotFoundException("Python exe not found.", _pythonExe);
        }

        if (!File.Exists(_pythonScript))
        {
            throw new FileNotFoundException("Python script not found.", _pythonScript);
        }

        var psi = new ProcessStartInfo
        {
            FileName = _pythonExe,
            // -u 关闭 Python 标准流缓冲 否则 C# ReadLine 可能等不到及时返回
            Arguments = $"-u \"{_pythonScript}\"",
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            StandardOutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            StandardErrorEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            // 工作目录设为脚本目录 便于 Python 侧按相对路径定位包和资源
            WorkingDirectory = Path.GetDirectoryName(_pythonScript) ?? Environment.CurrentDirectory
        };

        _process = new Process
        {
            StartInfo = psi,
            EnableRaisingEvents = true
        };

        _process.ErrorDataReceived += (_, e) =>
        {
            if (!string.IsNullOrWhiteSpace(e.Data))
            {
                Console.WriteLine("[PYTHON] " + e.Data);
            }
        };

        var startWatch = Stopwatch.StartNew();

        if (!_process.Start())
        {
            throw new Exception("Failed to start Python process.");
        }

        _process.BeginErrorReadLine();

        _stdin = _process.StandardInput;
        _stdout = _process.StandardOutput;
        _stdin.AutoFlush = true;
        var startedProcess = _process;
        _transport = new JsonLineTransport(_stdin, _stdout, () =>
        {
            if (!startedProcess.HasExited) startedProcess.Kill();
            startedProcess.WaitForExit(1000);
        });

        startWatch.Stop();

        Console.WriteLine($"Python process started. C# 启动进程耗时：{startWatch.Elapsed.TotalMilliseconds:F2} ms");
    }

    // 负责 C# 和 Python 的通信
    private TResponse Send<TRequest, TResponse>(TRequest request, out double roundtripMs,
        IDisposable? requestLifetime = null)
    {
        if (_transport == null) throw new InvalidOperationException("Python process is not ready.");
        string json = JsonSerializer.Serialize(request, _jsonOptions);
        var watch = Stopwatch.StartNew();
        int timeout = request is ReleaseRequest ? Math.Min(RequestTimeoutMilliseconds, 5000) : RequestTimeoutMilliseconds;
        string responseJson = _transport.Send(json, timeout, requestLifetime);
        roundtripMs = watch.Elapsed.TotalMilliseconds;
        return JsonSerializer.Deserialize<TResponse>(responseJson, _jsonOptions)
            ?? throw new InvalidDataException("Python returned an empty response.");
    }

    private static bool IsExplicitPath(string path)
    {
        return Path.IsPathRooted(path) ||
            path.Contains(Path.DirectorySeparatorChar) ||
            path.Contains(Path.AltDirectorySeparatorChar);
    }

    // 保存 C# 耗时信息
    private static void AddCsharpTimings(DetectResponse response)
    {
        // 同时写回独立字段和 timings 字典 便于调用方按对象属性或统一时间线两种方式读取
        response.Timings["csharp_image_read_ms"] = response.CsharpImageReadMs;
        response.Timings["csharp_shared_memory_write_ms"] = response.CsharpSharedMemoryWriteMs;
        response.Timings["csharp_roundtrip_ms"] = response.CsharpRoundtripMs;
        response.Timings["csharp_total_ms"] = response.CsharpTotalMs;
    }

    // 检查 Python 是否已经初始化
    private void EnsureInitialized()
    {
        if (!_initialized || _process == null || _process.HasExited)
        {
            throw new InvalidOperationException("Python detector is not initialized. Call Init first.");
        }
    }

    // 初始化算法请求
    private static InitAlgorithmRequest CreateInitAlgorithmRequest(string stage, string configPath)
    {
        if (string.IsNullOrWhiteSpace(stage))
        {
            throw new ArgumentException("Algorithm stage is empty.", nameof(stage));
        }

        string normalizedStage = stage.Trim().ToLowerInvariant();
        if (normalizedStage is not ("board" or "2d"))
        {
            throw new ArgumentException(
                $"Unsupported algorithm stage: {stage}. Use board for per-piece 2D/3D, or 2d.",
                nameof(stage));
        }

        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                "Bridge config manifest not found.", configPath);
        }

        string manifestText = File.ReadAllText(configPath, Encoding.UTF8);
        using JsonDocument manifestDocument = JsonDocument.Parse(manifestText);
        JsonElement manifest = manifestDocument.RootElement;
        if (manifest.ValueKind != JsonValueKind.Object
            || !manifest.TryGetProperty("config_2d", out JsonElement config2DElement)
            || !manifest.TryGetProperty("config_3d", out JsonElement config3DElement)
            || config2DElement.ValueKind != JsonValueKind.String
            || config3DElement.ValueKind != JsonValueKind.String)
        {
            throw new InvalidDataException(
                $"'config_2d' and 'config_3d' paths are required in {configPath}.");
        }

        string config2DPath = ResolveConfigRelativePathFrom(config2DElement.GetString()!, configPath);
        string config3DPath = ResolveConfigRelativePathFrom(config3DElement.GetString()!, configPath);
        if (!File.Exists(config2DPath) || !File.Exists(config3DPath))
        {
            throw new FileNotFoundException(
                $"Stage configuration file not found: {(!File.Exists(config2DPath) ? config2DPath : config3DPath)}");
        }

        string? inspectionPath = null;
        if (manifest.TryGetProperty("config_inspection", out JsonElement policyPath))
        {
            if (policyPath.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(policyPath.GetString()))
                throw new InvalidDataException("'config_inspection' must be a non-empty path.");
            inspectionPath = ResolveConfigRelativePathFrom(policyPath.GetString()!, configPath);
        }
        using JsonDocument? inspection = inspectionPath == null
            ? null
            : JsonDocument.Parse(File.ReadAllText(inspectionPath, Encoding.UTF8));
        if (inspection != null && inspection.RootElement.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException("Inspection configuration must be an object.");
        string configText;
        using (JsonDocument config2D = JsonDocument.Parse(File.ReadAllText(config2DPath, Encoding.UTF8)))
        using (JsonDocument config3D = JsonDocument.Parse(File.ReadAllText(config3DPath, Encoding.UTF8)))
        {
            using var merged = new MemoryStream();
            using (var writer = new Utf8JsonWriter(merged))
            {
                writer.WriteStartObject();
                foreach (JsonProperty property in config2D.RootElement.EnumerateObject())
                {
                    if (inspection == null || !property.NameEquals("inspection"))
                        property.WriteTo(writer);
                }
                if (normalizedStage == "board")
                {
                    foreach (JsonProperty property in config3D.RootElement.EnumerateObject())
                    {
                        if (!property.NameEquals("model_path") && (inspection == null || !property.NameEquals("inspection")))
                            property.WriteTo(writer);
                    }
                }
                if (inspection != null)
                {
                    writer.WritePropertyName("inspection");
                    inspection.RootElement.WriteTo(writer);
                }
                writer.WriteEndObject();
            }
            configText = Encoding.UTF8.GetString(merged.ToArray());
        }

        using JsonDocument stageConfig = JsonDocument.Parse(configText);
        JsonElement stageRoot = stageConfig.RootElement;
        bool hasModelPaths = stageRoot.TryGetProperty(
            "model_path", out JsonElement modelPathElement);
        if (normalizedStage is ("board" or "2d")
            && (!hasModelPaths || modelPathElement.ValueKind != JsonValueKind.Object))
        {
            throw new InvalidDataException($"'model_path' object is required in {config2DPath}.");
        }

        var modelPaths = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        IEnumerable<JsonProperty> modelProperties = hasModelPaths
            ? modelPathElement.EnumerateObject().ToArray()
            : Array.Empty<JsonProperty>();
        foreach (JsonProperty model in modelProperties)
        {
            if (model.Value.ValueKind != JsonValueKind.String
                || string.IsNullOrWhiteSpace(model.Value.GetString()))
            {
                throw new InvalidDataException(
                    $"Every entry in 'model_path' must contain a non-empty path.");
            }
            string resolvedPath = ResolveConfigRelativePathFrom(
                model.Value.GetString()!, config2DPath);
            if (!File.Exists(resolvedPath))
            {
                throw new FileNotFoundException(
                    $"Model file for '{model.Name}' not found.", resolvedPath);
            }
            modelPaths[model.Name] = resolvedPath;
        }

        if (normalizedStage is ("board" or "2d") && !modelPaths.ContainsKey("YOLO"))
        {
            throw new InvalidDataException("model_path.YOLO is required.");
        }

        return new InitAlgorithmRequest
        {
            Stage = normalizedStage,
            ModelPath = modelPaths,
            ConfigPath = config2DPath,
            Config3DPath = normalizedStage == "board" ? config3DPath : null,
            ConfigText = configText
        };
    }

    private static string ResolveConfigRelativePathFrom(string path, string configPath)
    {
        if (Path.IsPathRooted(path))
        {
            return Path.GetFullPath(path);
        }

        string baseDirectory = Path.GetDirectoryName(configPath)
            ?? Environment.CurrentDirectory;
        return Path.GetFullPath(Path.Combine(baseDirectory, path));
    }

    private static string ResolveConfigPath(string pythonScript)
    {
        string scriptPath = Path.GetFullPath(pythonScript);
        string scriptDirectory = Path.GetDirectoryName(scriptPath)
            ?? Environment.CurrentDirectory;
        string? projectRoot = Directory.GetParent(scriptDirectory)?.Parent?.FullName;

        string[] candidates =
        {
            projectRoot == null ? string.Empty : Path.Combine(projectRoot, "config", "config.json"),
            Path.Combine(Environment.CurrentDirectory, "config", "config.json"),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "config", "config.json")
        };

        foreach (string candidate in candidates)
        {
            if (!string.IsNullOrWhiteSpace(candidate) && File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }

        return Path.GetFullPath(candidates[0]);
    }

    //创建共享内存名称
    private string CreateSharedMemoryName(string key)
    {
        return $"bridge_{_bridgeId}_{Sanitize(key)}";
    }

    // 创建/复用共享内存
    private SharedMemoryBuffer CreateSharedMemoryBuffer(string key, int dataSize)
    {
        string name = CreateSharedMemoryName(key) + "_" + Guid.NewGuid().ToString("N");
        var mapping = MemoryMappedFile.CreateNew(name, dataSize);
        try
        {
            return new SharedMemoryBuffer(name, dataSize, mapping,
                mapping.CreateViewAccessor(0, dataSize, MemoryMappedFileAccess.Write));
        }
        catch { mapping.Dispose(); throw; }
    }

    private IDisposable EnterLifecycleOperation()
    {
        _lifecycleLock.EnterWriteLock();
        return new OperationLease(() => _lifecycleLock.ExitWriteLock());
    }

    private IDisposable EnterDetection()
    {
        if (!_inflight.Wait(RequestTimeoutMilliseconds))
            throw new TimeoutException("Maximum in-flight detector requests reached.");
        try
        {
            _lifecycleLock.EnterReadLock();
            try
            {
                EnsureInitialized();
                if (_closing || _disposed) throw new ObjectDisposedException(nameof(PythonBridge));
                return new OperationLease(() =>
                {
                    _lifecycleLock.ExitReadLock();
                    _inflight.Release();
                });
            }
            catch { _lifecycleLock.ExitReadLock(); throw; }
        }
        catch { _inflight.Release(); throw; }
    }

    private sealed class OperationLease : IDisposable
    {
        private Action? _release;
        public OperationLease(Action release) { _release = release; }
        public void Dispose() { Interlocked.Exchange(ref _release, null)?.Invoke(); }
    }

    private static IDisposable? HoldRequestMappings<TRequest>(TRequest request)
    {
        IEnumerable<string> names = request switch
        {
            Detect2DRequest image => new[] { image.ShmName },
            Detect3DRequest depth => new[] { depth.ShmName },
            _ => Array.Empty<string>()
        };
        var mappings = new List<MemoryMappedFile>();
        try
        {
            foreach (string name in names.Distinct())
                mappings.Add(MemoryMappedFile.OpenExisting(name, MemoryMappedFileRights.Read));
            return new OperationLease(() => { foreach (var mapping in mappings) mapping.Dispose(); });
        }
        catch { foreach (var mapping in mappings) mapping.Dispose(); throw; }
    }

    // 替换非法共享内存名称
    private static string Sanitize(string value)
    {
        var builder = new StringBuilder(value.Length);

        foreach (char ch in value)
        {
            builder.Append(char.IsLetterOrDigit(ch) ? ch : '_');
        }

        return builder.ToString();
    }

    // 对象销毁时清理
    public void Dispose()
    {
        _disposed = true;
        try
        {
            Release();
        }
        catch
        {
            // Dispose 不向外抛清理失败 避免掩盖调用方原始异常
        }
    }

    // 清理PythonBridge占用资源
    private void CleanupProcessObjects()
    {
        _transport?.Dispose();
        _transport = null;

        _stdin?.Dispose();
        _stdout?.Dispose();
        _process?.Dispose();

        _stdin = null;
        _stdout = null;
        _process = null;
        _initialized = false;
    }

    private sealed class SharedMemoryBuffer : IDisposable
    {
        public SharedMemoryBuffer(
            string name,
            int capacity,
            MemoryMappedFile memoryMappedFile,
            MemoryMappedViewAccessor accessor)
        {
            Name = name;
            Capacity = capacity;
            MemoryMappedFile = memoryMappedFile;
            Accessor = accessor;
        }

        public string Name { get; }

        public int Capacity { get; }

        public MemoryMappedFile MemoryMappedFile { get; }

        public MemoryMappedViewAccessor Accessor { get; }

        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            Accessor.Dispose();
            MemoryMappedFile.Dispose();
        }
    }

    private sealed class BufferGroup : IDisposable
    {
        private List<SharedMemoryBuffer>? _buffers;
        public BufferGroup(IEnumerable<SharedMemoryBuffer> buffers) { _buffers = buffers.ToList(); }
        public void Dispose()
        {
            var buffers = Interlocked.Exchange(ref _buffers, null);
            if (buffers == null) return;
            foreach (var buffer in buffers) buffer.Dispose();
        }
    }
}

public sealed class Bridge2D
{
    private readonly PythonBridge _bridge;

    internal Bridge2D(PythonBridge bridge)
    {
        _bridge = bridge;
    }

    public Task<DetectResponse> DetectPieceFileAsync(string boardId, int pieceId, string imagePath,
        object? threshold = null)
        => Task.Run(() => DetectPieceFile(boardId, pieceId, imagePath, threshold));

    public Task<DetectResponse> DetectPieceSharedMemoryAsync(
        string boardId, int pieceId, string shmName, int imageSize,
        int width, int height, int channels, string imageFormat = "rgb8", object? threshold = null)
        => Task.Run(() => DetectPieceSharedMemory(boardId, pieceId, shmName,
            imageSize, width, height, channels, imageFormat, threshold));

    /// <summary>Read-only BGR view with optional padding/offset. Keep the mapping alive until return.</summary>
    /// <remarks>ImageSize is the available byte count from ImageOffset; retain the mapping unchanged until the returned task completes.</remarks>
    public DetectResponse DetectPieceSharedMemoryLayout(
        string boardId, int pieceId, string shmName, int imageSize,
        int width, int height, int channels, int stride, long offset = 0, object? threshold = null)
    {
        if (string.IsNullOrWhiteSpace(boardId) || pieceId <= 0)
            throw new ArgumentException("Board ID and positive piece ID are required.");
        if (string.IsNullOrWhiteSpace(shmName) || width <= 0 || height <= 0 || channels != 3
            || stride < checked(width * channels) || offset < 0
            || imageSize < checked((long)stride * (height - 1) + width * channels))
            throw new ArgumentException("Invalid BGR shared-memory layout.");
        return _bridge.Detect2DSharedMemory(new Detect2DRequest
        {
            Cmd = "detect_2d_piece", BoardId = boardId, PieceId = pieceId,
            ShmName = shmName, ImageSize = imageSize, ImageWidth = width,
            ImageHeight = height, ImageChannels = channels, ImageFormat = "bgr8",
            ImageStride = stride, ImageOffset = offset, Threshold = threshold
        });
    }

    public Task<DetectResponse> DetectPieceSharedMemoryLayoutAsync(
        string boardId, int pieceId, string shmName, int imageSize,
        int width, int height, int channels, int stride, long offset = 0, object? threshold = null)
        => Task.Run(() => DetectPieceSharedMemoryLayout(
            boardId, pieceId, shmName, imageSize, width, height, channels, stride, offset, threshold));

    // 图像文件调试接口
    public DetectResponse Detect2DFile(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path is empty.", nameof(imagePath));
        }

        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException("Image file not found.", imagePath);
        }

        return _bridge.Detect2DFile(imagePath);
    }

    // 新流程：每次提交一张明确属于某块板、某一片的 2D 图片。
    public DetectResponse DetectPieceFile(
        string boardId,
        int pieceId,
        string imagePath,
        object? threshold = null)
    {
        if (string.IsNullOrWhiteSpace(boardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(boardId));
        }
        if (pieceId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pieceId));
        }
        if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
        {
            throw new FileNotFoundException("2D image file not found.", imagePath);
        }
        return _bridge.Detect2DPieceFile(boardId, pieceId, imagePath, threshold);
    }

    // 新流程：采集端已经把单片图像写入共享内存，只传元数据和板/片号。
    public DetectResponse DetectPieceSharedMemory(
        string boardId,
        int pieceId,
        string shmName,
        int imageSize,
        int width,
        int height,
        int channels,
        string imageFormat = "rgb8",
        object? threshold = null)
    {
        if (string.IsNullOrWhiteSpace(boardId))
        {
            throw new ArgumentException("Board ID is empty.", nameof(boardId));
        }
        if (pieceId <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pieceId));
        }
        ValidateRawImageMetadata(shmName, imageSize, width, height, channels, imageFormat);
        return _bridge.Detect2DSharedMemory(new Detect2DRequest
        {
            Cmd = "detect_2d_piece",
            BoardId = boardId,
            PieceId = pieceId,
            ShmName = shmName,
            ImageSize = imageSize,
            ImageFormat = imageFormat,
            ImageWidth = width,
            ImageHeight = height,
            ImageChannels = channels,
            Threshold = threshold
        });
    }


    // 共享内存入口 该入口给外部采集程序使用 图像已在共享内存里
    public DetectResponse Detect2DSharedMemory(
        string shmName,
        int imageSize,
        int width,
        int height,
        int channels,
        string imageFormat = "rgb8")
    {
        ValidateRawImageMetadata(shmName, imageSize, width, height, channels, imageFormat);

        return _bridge.Detect2DSharedMemory(new Detect2DRequest
        {
            Cmd = "detect",
            ShmName = shmName,
            ImageSize = imageSize,
            ImageFormat = imageFormat,
            ImageWidth = width,
            ImageHeight = height,
            ImageChannels = channels
        });
    }

    private static void ValidateRawImageMetadata(
        string shmName,
        int imageSize,
        int width,
        int height,
        int channels,
        string imageFormat)
    {
        if (string.IsNullOrWhiteSpace(shmName))
        {
            throw new ArgumentException("Shared memory name is empty.", nameof(shmName));
        }
        if (imageSize <= 0)
        {
            throw new ArgumentException("Image size must be greater than 0.", nameof(imageSize));
        }
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException("Image dimensions must be greater than 0.");
        }
        if (channels != 1 && channels != 3)
        {
            throw new ArgumentException("Only 1- or 3-channel images are supported.", nameof(channels));
        }
        if (string.Equals(imageFormat, "rgb8", StringComparison.OrdinalIgnoreCase)
            || string.Equals(imageFormat, "bgr8", StringComparison.OrdinalIgnoreCase)
            || string.Equals(imageFormat, "gray8", StringComparison.OrdinalIgnoreCase)
            || string.Equals(imageFormat, "raw", StringComparison.OrdinalIgnoreCase))
        {
            int expectedSize = checked(width * height * channels);
            if (imageSize != expectedSize)
            {
                throw new ArgumentException(
                    $"Image size mismatch. Expected {expectedSize}, actual {imageSize}.");
            }
        }
    }

}

public sealed class Bridge3D
{
    private readonly PythonBridge _bridge;

    internal Bridge3D(PythonBridge bridge)
    {
        _bridge = bridge;
    }

    private Task<DetectResponse> DetectDepthAsync(ushort[] depth, int width, int height, Detect3DRequest request)
    {
        int size = PackedDepthInput.ByteCount(width, height);
        if (depth == null || depth.LongLength * sizeof(ushort) != size)
            throw new ArgumentException("3D array length must equal width * height.", nameof(depth));
        var snapshot = (ushort[])depth.Clone();
        if (request.PitchCorrection != null)
            request.PitchCorrection = new Dictionary<string, object>(request.PitchCorrection);
        return Task.Run(() => _bridge.Detect3DValues(snapshot, request, width, height));
    }

    /// <summary>Submit an already separated depth image for one piece.</summary>
    public DetectResponse DetectPiece(string boardId, int pieceId, ushort[] depth, int width, int height,
        Dictionary<string, object>? correction = null)
        => _bridge.Detect3DValues(depth, PieceRequest(boardId, pieceId, correction), width, height);

    public Task<DetectResponse> DetectPieceAsync(string boardId, int pieceId, ushort[] depth, int width, int height,
        Dictionary<string, object>? correction = null)
        => DetectDepthAsync(depth, width, height, PieceRequest(boardId, pieceId, correction));

    public DetectResponse DetectPieceFile(string boardId, int pieceId, string path,
        Dictionary<string, object>? correction = null)
        => _bridge.DetectStreamed3DFile(path, PieceRequest(boardId, pieceId, correction));

    public Task<DetectResponse> DetectPieceFileAsync(string boardId, int pieceId, string path,
        Dictionary<string, object>? correction = null)
    {
        var metadata = correction == null ? null : new Dictionary<string, object>(correction);
        return Task.Run(() => DetectPieceFile(boardId, pieceId, path, metadata));
    }

    public DetectResponse DetectPieceSharedMemory(string boardId, int pieceId, string shmName, int dataSize,
        int width, int height, Dictionary<string, object>? correction = null)
    {
        var request = PieceRequest(boardId, pieceId, correction);
        request.ShmName = shmName;
        request.DataSize = dataSize;
        request.ImageWidth = width;
        request.ImageHeight = height;
        return _bridge.DetectStreamed3DSharedMemory(request);
    }

    public Task<DetectResponse> DetectPieceSharedMemoryAsync(string boardId, int pieceId, string shmName, int dataSize,
        int width, int height, Dictionary<string, object>? correction = null)
    {
        var metadata = correction == null ? null : new Dictionary<string, object>(correction);
        return Task.Run(() => DetectPieceSharedMemory(boardId, pieceId, shmName, dataSize, width, height, metadata));
    }

    private static Detect3DRequest PieceRequest(string boardId, int pieceId, Dictionary<string, object>? correction)
    {
        if (string.IsNullOrWhiteSpace(boardId))
            throw new ArgumentException("Board ID is empty.", nameof(boardId));
        if (pieceId <= 0)
            throw new ArgumentOutOfRangeException(nameof(pieceId));
        return new Detect3DRequest {
            Cmd = "detect_3d_piece", BoardId = boardId, PieceId = pieceId, PitchCorrection = correction };
    }
}

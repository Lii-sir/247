using Bridge;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

static class Program
{
    static int Main(string[] args)
    {
        try
        {
            string root = Path.GetFullPath(args[0]);
            string python = Path.Combine(root, ".venv", "Scripts", "python.exe");
            bool real = args.Length > 1 && args[1] == "--real";
            string fixture = real
                ? Path.Combine(root, "python", "transport", "bridge_server.py")
                : Path.Combine(root, "tests", "fake_concurrent_server.py");
            using var bridge = new PythonBridge(python, fixture, 13);
            if (real)
            {
                Environment.SetEnvironmentVariable("BRIDGE_BOARD_TIMEOUT_SECONDS", "900");
                bridge.RequestTimeoutMilliseconds = 900000;
            }
            bridge.Init();
            if (real)
            {
                bridge.InitAlgorithm("board");
                string[] files = Directory.GetFiles(Path.Combine(root, "demodatasets", "2D"), "*.png")
                    .OrderBy(path => path, StringComparer.OrdinalIgnoreCase).ToArray();
                if (files.Length != 10) throw new InvalidDataException($"Expected ten sample 2D images, found {files.Length}.");
                string depthFile = Directory.GetFiles(Path.Combine(root, "demodatasets", "3D"), "*.png").Single();
                string boardId = "real-" + Guid.NewGuid().ToString("N");
                var scan = bridge.ThreeD.DetectBoardFileAsync(boardId, depthFile);
                Thread.Sleep(100);
                var tasks = files.Select((file, index) => bridge.TwoD.DetectPieceFileAsync(boardId, index + 1, file)).ToArray();
                Check(Task.WaitAll(tasks, 900000), "Real 2D sample requests timed out.");
                Check(scan.Wait(900000), "Real board registration/measurement timed out.");
                Check(tasks.All(task => task.Result.Ok), "At least one real 2D sample failed.");
                Check(scan.Result.Ok && scan.Result.ResultsBoard?.Pieces.Count == 10,
                    "Real 3D board result did not contain ten pieces.");
                Console.WriteLine($"REAL PASS: ten 2D samples + one 3D board; board_ok={scan.Result.ResultsBoard!.IsOk}.");
                return 0;
            }
            ThreadPool.SetMinThreads(16, 16);
            ushort[] depthValues = Enumerable.Range(0, 300).Select(i =>
                i / 3 % 10 < 5 ? (ushort)(33000 + i) : (ushort)0).ToArray();
            depthValues[0] = 1; depthValues[1] = 256; depthValues[2] = 32768; depthValues[3] = 65535;
            byte[] depthBytes = new byte[600];
            Buffer.BlockCopy(depthValues, 0, depthBytes, 0, depthBytes.Length);
            using var depth = Mapping(depthBytes);
            var inputs = Enumerable.Range(1, 10).Select(pid => Mapping(new byte[] { (byte)pid, 2, 3 })).ToArray();
            try
            {
                // A depth request arrives first, yet all ten camera calls must keep progressing.
                var scan = bridge.ThreeD.DetectBoardSharedMemoryAsync("test-board", depth.Name, 600, 3, 100);
                Thread.Sleep(50);
                Check(!scan.IsCompleted, "3D must wait for matching pieces, not fail early.");
                var images = inputs.Select((mapping, index) =>
                    bridge.TwoD.DetectPieceSharedMemoryAsync("test-board", index + 1,
                        mapping.Name, 3, 1, 1, 3, "bgr8")).ToArray();
                Check(Task.WaitAll(images, 10000), "2D tasks did not finish (serialization/deadlock).");
                Check(scan.Wait(10000), "3D aggregation did not finish.");
                Check(images.All(task => task.Result.Ok), "2D failed.");
                Check(images.Select(task => task.Result.PieceId).OrderBy(x => x)
                    .SequenceEqual(Enumerable.Range(1, 10).Select(x => (int?)x)), "Replies were misrouted.");
                Check(images.Select(task => task.Result.RequestId).Distinct().Count() == 10,
                    "Requests do not have unique correlation IDs.");
                Check(scan.Result.Ok && scan.Result.ResultsBoard?.Pieces.Count == 10,
                    "Final board must contain exactly ten matching pieces.");
                Check(scan.Result.ResultsBoard?.DepthCoordinateSpace == "board_3d",
                    "Board depth coordinate tag was lost during DTO mapping.");
                Check(scan.Result.ResultsBoard?.PitchCorrection != null &&
                    scan.Result.ResultsBoard.PitchCorrection.TryGetValue("applied", out var applied) && !applied.GetBoolean(),
                    "Pitch correction metadata was lost during DTO mapping.");

                var orphan = bridge.ThreeD.DetectBoardSharedMemoryAsync("orphan-board", depth.Name, 600, 3, 100);
                Thread.Sleep(50);
                Check(bridge.ReleaseBoard("orphan-board").Ok, "Explicit release failed.");
                Check(orphan.Wait(5000) && !orphan.Result.Ok, "Released pending request remained blocked.");

                // Thirteen simultaneous callers all use the same bridge/reader.
                var concurrent = Enumerable.Range(1, 13).Select(pid => Task.Run(() =>
                    bridge.TwoD.DetectPieceSharedMemory("parallel-" + ((pid - 1) / 10), (pid - 1) % 10 + 1,
                        inputs[0].Name, 3, 1, 1, 3, "bgr8"))).ToArray();
                Check(Task.WaitAll(concurrent, 10000) && concurrent.All(t => t.Result.Ok),
                    "Thirteen simultaneous requests failed.");
                bridge.ReleaseBoard("parallel-0");
                bridge.ReleaseBoard("parallel-1");

                string temp = Path.Combine(Path.GetTempPath(), "bridge-stream-" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(temp);
                try
                {
                    string[] files = Enumerable.Range(1, 10).Select(pid =>
                    {
                        string file = Path.Combine(temp, pid.ToString("D2") + ".png");
                        File.WriteAllBytes(file, new byte[] { (byte)pid, 2, 3 });
                        return file;
                    }).ToArray();
                    string depthFile = Path.Combine(temp, "depth.png");
                    var source = BitmapSource.Create(3, 100, 96, 96, PixelFormats.Gray16, null, depthValues, 6);
                    var encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(source));
                    using (var file = File.Create(depthFile)) encoder.Save(file);
                    DetectResponse joined = bridge.Board.DetectStreamingFilesAsync(
                        files, depthFile, "high-level-board").GetAwaiter().GetResult();
                    Check(joined.Ok && joined.ResultsBoard?.Pieces.Count == 10,
                        "High-level 2D/3D orchestration failed.");
                }
                finally { Directory.Delete(temp, recursive: true); }

                var correction = new Dictionary<string, object> {
                    ["applied"] = true, ["model"] = "smartray_pitch", ["angle_deg"] = 6.0,
                    ["scan_axis"] = "rows", ["scan_pitch_mm"] = .02,
                    ["z_step_mm"] = .00105, ["gray_offset"] = 32768.0,
                    ["z_origin_mm"] = 0.0, ["transform_direction"] = "recover_ideal"
                };
                var sdk = bridge.ThreeD.DetectBoardAsync("sdk-array", depthValues, 3, 100, correction);
                correction["angle_deg"] = 59.0;
                // The SDK is allowed to overwrite the original after submission.
                Array.Clear(depthValues, 0, depthValues.Length);
                var sdkImages = Enumerable.Range(1, 10).Select(pid =>
                    bridge.TwoD.DetectPieceSharedMemoryAsync("sdk-array", pid, inputs[0].Name, 3, 1, 1, 3, "bgr8")).ToArray();
                Check(Task.WaitAll(sdkImages, 10000) && sdk.Wait(10000) && sdk.Result.Ok,
                    "SDK ushort array submission or its synchronous snapshot failed.");
                Check(sdk.Result.ResultsBoard?.DepthCoordinateSpace == "pitch_corrected_board_3d",
                    "Externally prepared depth coordinate metadata was lost.");
                Check(sdk.Result.ResultsBoard!.PitchCorrection!["angle_deg"].GetDouble() == 6.0,
                    "Async correction metadata was not snapshotted before returning.");
                Check(sdk.Result.Timings != null && !sdk.Result.Timings.ContainsKey("smartray_pitch_ms"),
                    "Detection time unexpectedly includes preprocessing.");
                bool rejected = false;
                try { bridge.ThreeD.DetectBoardSharedMemory("bad-size", depth.Name, 599, 3, 100); }
                catch (ArgumentException) { rejected = true; }
                Check(rejected, "Raw shared depth size mismatch was not rejected.");

                bridge.RequestTimeoutMilliseconds = 150;
                bool timedOut = false;
                try { bridge.ThreeD.DetectBoardSharedMemory("never-complete", depth.Name, 600, 3, 100); }
                catch (TimeoutException) { timedOut = true; }
                Check(timedOut, "Request timeout did not terminate a stalled worker.");
                bridge.RequestTimeoutMilliseconds = 120000;
                bridge.Init();
                Check(bridge.TwoD.DetectPieceSharedMemory("restart", 1,
                    inputs[0].Name, 3, 1, 1, 3, "bgr8").Ok, "Restart after timeout failed.");
                bridge.ReleaseBoard("restart");
            }
            finally { foreach (var input in inputs) input.Dispose(); }
            Console.WriteLine("PASS: 3D-first/10 camera tasks, correlation, 13 concurrent callers, release, timeout/restart.");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }

    static TestMapping Mapping(byte[] bytes) => new(bytes);

    sealed class TestMapping : IDisposable
    {
        public string Name { get; } = "bridge-test-" + Guid.NewGuid().ToString("N");
        readonly MemoryMappedFile _mapping;
        public TestMapping(byte[] bytes)
        {
            _mapping = MemoryMappedFile.CreateNew(Name, bytes.Length);
            using var view = _mapping.CreateViewAccessor();
            view.WriteArray(0, bytes, 0, bytes.Length);
        }
        public void Dispose() => _mapping.Dispose();
    }
}

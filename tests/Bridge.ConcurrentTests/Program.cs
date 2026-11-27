using Bridge;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

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
            bridge.Init();
            bridge.InitAlgorithm("board");
            if (real)
            {
                if (args.Length != 4)
                    throw new ArgumentException("Usage: Bridge.ConcurrentTests.exe <root> --real <2d> <depth-piece>");
                Check(bridge.TwoD.DetectPieceFile("sample", 1, args[2]).Ok, "Sample 2D input failed.");
                Check(bridge.ThreeD.DetectPieceFile("sample", 1, args[3]).ResultsPiece?.PieceId == 1, "Sample 3D input failed.");
                return 0;
            }
            CheckProductConfiguration(bridge, root);

            ThreadPool.SetMinThreads(16, 16);
            ushort[] values = Enumerable.Range(0, 30).Select(i => (ushort)(33000 + i)).ToArray();
            values[0] = 1; values[1] = 256; values[2] = 32768; values[3] = 65535;
            byte[] bytes = new byte[60];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            using var depth = Mapping(bytes);
            using var image = Mapping(new byte[] { 1, 2, 3 });

            var ids = Enumerable.Range(0, 13).Select(index => (Board: "board-" + index / 10, Piece: index % 10 + 1)).ToArray();
            var twoD = ids.Select(id => bridge.TwoD.DetectPieceSharedMemoryAsync(
                id.Board, id.Piece, image.Name, 3, 1, 1, 3, "bgr8")).ToArray();
            Check(Task.WaitAll(twoD, 10000) && twoD.All(task => task.Result.Ok), "Parallel 2D requests failed.");
            Check(twoD.Select(task => task.Result.RequestId).Distinct().Count() == ids.Length, "Request IDs were reused.");

            var threeD = ids.Select(id => bridge.ThreeD.DetectPieceSharedMemoryAsync(
                id.Board, id.Piece, depth.Name, 60, 3, 10)).ToArray();
            Check(Task.WaitAll(threeD, 10000) && threeD.All(task => task.Result.Ok), "Parallel 3D requests failed.");
            for (int index = 0; index < threeD.Length; index++)
            {
                var result = threeD[index].Result;
                Check(result.BoardId == ids[index].Board && result.ResultsPiece?.PieceId == ids[index].Piece,
                    "3D result was routed to the wrong board or piece.");
            }

            var correction = new Dictionary<string, object> {
                ["applied"] = true, ["model"] = "smartray_pitch", ["angle_deg"] = 6.0,
                ["scan_axis"] = "rows", ["scan_pitch_mm"] = .02, ["z_step_mm"] = .00105,
                ["gray_offset"] = 32768.0, ["z_origin_mm"] = 0.0, ["transform_direction"] = "recover_ideal"
            };
            Check(bridge.TwoD.DetectPieceSharedMemory("sdk", 4, image.Name, 3, 1, 1, 3, "bgr8").Ok, "Array fixture 2D failed.");
            var sdk = bridge.ThreeD.DetectPieceAsync("sdk", 4, values, 3, 10, correction);
            Array.Clear(values, 0, values.Length);
            correction["angle_deg"] = 59.0;
            Check(sdk.Wait(5000) && sdk.Result.Ok &&
                sdk.Result.ResultsPiece is { PieceId: 4, DepthCoordinateSpace: "pitch_corrected_piece_3d", DepthEndRow: 10 },
                "Array snapshot or coordinate metadata was lost.");
            Check(sdk.Result.ResultsPiece!.PitchCorrection!["angle_deg"].GetDouble() == 6.0,
                "Correction metadata was not snapshotted.");

            var skipped = bridge.ThreeD.DetectPieceSharedMemory("missing", 1, depth.Name, 60, 3, 10);
            Check(skipped.Ok && skipped.ResultsPiece is { IsOk: false, Skip3DReason: "missing_2d_result" },
                "Missing 2D did not return a skipped NG piece.");
            bool rejected = false;
            try { bridge.ThreeD.DetectPieceSharedMemory("invalid", 1, depth.Name, 59, 3, 10); }
            catch (ArgumentException) { rejected = true; }
            Check(rejected, "Raw shared depth size mismatch was accepted.");

            var pending = bridge.TwoD.DetectPieceSharedMemoryAsync("release-pending", 1, image.Name, 3, 1, 1, 3, "bgr8");
            Thread.Sleep(100);
            Check(bridge.ReleaseBoard("release-pending").Ok, "Explicit release failed.");
            Check(pending.Wait(5000) && !pending.Result.Ok, "Released request did not fail.");

            bridge.RequestTimeoutMilliseconds = 150;
            bool timedOut = false;
            try { bridge.TwoD.DetectPieceSharedMemory("never-complete", 1, image.Name, 3, 1, 1, 3, "bgr8"); }
            catch (TimeoutException) { timedOut = true; }
            Check(timedOut, "Request timeout did not terminate a stalled worker.");
            bridge.RequestTimeoutMilliseconds = 120000;
            bridge.Init();
            Check(bridge.TwoD.DetectPieceSharedMemory("restart", 1, image.Name, 3, 1, 1, 3, "bgr8").Ok,
                "Restart after timeout failed.");
            bridge.ReleaseBoard("restart");
            Console.WriteLine("PASS: per-piece concurrency, metadata, missing 2D, release, and timeout/restart.");
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

    static void CheckProductConfiguration(PythonBridge bridge, string root)
    {
        var fixture = Directory.CreateDirectory(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "bridge-product-" + Guid.NewGuid().ToString("N")));
        try
        {
            string config = Path.Combine(root, "config", "products", "normal");
            File.WriteAllText(Path.Combine(fixture.FullName, "inspection.json"),
                "{\"defect_sequence\":[\"wire_height_3d\"],\"skip_3d_on_2d_ng\":true}");
            string manifest = Path.Combine(fixture.FullName, "config.json");
            File.WriteAllText(manifest, JsonSerializer.Serialize(new {
                config_2d = Path.Combine(config, "2D", "config.json"),
                config_3d = Path.Combine(config, "3D", "config.json"),
                config_inspection = "inspection.json"
            }));
            Check(bridge.InitAlgorithm("board", manifest).Message == "selected product",
                "Product-specific inspection settings were not loaded.");
            Check(bridge.InitAlgorithm("board").Message == "default product",
                "The default manifest did not return after using an explicit one.");
        }
        finally
        {
            fixture.Delete(recursive: true);
        }
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

using System;
using System.Globalization;
using System.IO;
using SmartRay.PitchCorrection;

internal static class PitchRunner
{
    private static double Number(string value) { return Double.Parse(value, CultureInfo.InvariantCulture); }
    private static void Require(bool condition) { if (!condition) throw new Exception("Contract check failed."); }
    private static void Reject(Action action)
    {
        try { action(); } catch (ArgumentException) { return; }
        throw new Exception("Invalid input was accepted.");
    }
    private static int Main(string[] args)
    {
        if (args.Length == 1 && args[0] == "contracts")
        {
            Reject(delegate { new PitchCorrector(null); });
            Reject(delegate { new PitchCorrector(new PitchOptions { AngleDegrees = 60, ScanPitchMm = .02 }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = 0 }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = Double.NaN }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = .02, ZStepMm = -1 }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = .02, GrayOffset = Double.PositiveInfinity }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = .02, ProjectionWorkers = 0 }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = .02, Axis = (ScanAxis)5 }); });
            Reject(delegate { new PitchCorrector(new PitchOptions { ScanPitchMm = .02, Direction = (TransformDirection)5 }); });
            PitchCorrector corrector = new PitchCorrector(new PitchOptions { ScanPitchMm = .02 });
            Reject(delegate { corrector.Correct(null, 1, 1); });
            Reject(delegate { corrector.Correct(new ushort[1], 0, 1); });
            Reject(delegate { corrector.Correct(new ushort[1], Int32.MaxValue, 2); });
            Reject(delegate { corrector.Correct(new ushort[1], 2, 1); });
            ushort[] array = new ushort[1];
            Reject(delegate { corrector.CorrectInto(array, array, 1, 1); });
            Reject(delegate { corrector.CorrectInto(array, null, 1, 1); });
            Reject(delegate { corrector.CorrectInto(array, new ushort[2], 1, 1); });
            Console.WriteLine("16 API rejection checks passed.");
            return 0;
        }
        int width = Int32.Parse(args[2]), height = Int32.Parse(args[3]);
        ushort[] input = new ushort[width * height];
        using (BinaryReader reader = new BinaryReader(File.OpenRead(args[0])))
            for (int i = 0; i < input.Length; ++i) input[i] = reader.ReadUInt16();
        ushort[] original = (ushort[])input.Clone();
        PitchOptions options = new PitchOptions
        {
            AngleDegrees = Number(args[4]), ScanPitchMm = Number(args[5]), ZStepMm = Number(args[6]),
            GrayOffset = Number(args[7]), ZOriginMm = Number(args[8]),
            Axis = (ScanAxis)Int32.Parse(args[9]), Direction = (TransformDirection)Int32.Parse(args[10]),
            ProjectionWorkers = Int32.Parse(args[11]), Enabled = args[12] == "1"
        };
        PitchCorrector correctorInstance = new PitchCorrector(options);
        // Changing the caller's options must not mutate an existing instance.
        options.AngleDegrees = Double.NaN;
        ushort[] actual = correctorInstance.Correct(input, width, height);
        ushort[] reused = new ushort[actual.Length];
        for (int i = 0; i < reused.Length; ++i) reused[i] = 65535;
        correctorInstance.CorrectInto(input, reused, width, height);
        for (int i = 0; i < input.Length; ++i)
        {
            Require(input[i] == original[i]);
            Require(actual[i] == reused[i]);
        }
        // Concurrent frame calls share only immutable tables, not output pixels.
        System.Threading.Tasks.Parallel.For(0, 3, delegate(int job)
        {
            ushort[] parallel = correctorInstance.Correct(input, width, height);
            ushort[] parallelReused = new ushort[input.Length];
            for (int i = 0; i < parallelReused.Length; ++i) parallelReused[i] = 65535;
            correctorInstance.CorrectInto(input, parallelReused, width, height);
            for (int i = 0; i < parallel.Length; ++i)
            {
                Require(parallel[i] == actual[i]);
                Require(parallelReused[i] == actual[i]);
            }
        });
        // Reusing a prefilled buffer for a different frame must clear old pixels.
        ushort[] zero = new ushort[input.Length];
        correctorInstance.CorrectInto(zero, reused, width, height);
        for (int i = 0; i < reused.Length; ++i) Require(reused[i] == 0);
        using (BinaryWriter writer = new BinaryWriter(File.Create(args[1])))
            foreach (ushort value in actual) writer.Write(value);
        return 0;
    }
}

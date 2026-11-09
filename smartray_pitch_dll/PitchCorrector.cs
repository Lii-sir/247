using System;
using System.Threading.Tasks;
using System.Reflection;

[assembly: AssemblyTitle("SmartRay Pitch Correction")]
[assembly: AssemblyVersion("1.0.0.0")]

namespace SmartRay.PitchCorrection
{
    public enum ScanAxis { Rows = 0, Columns = 1 }
    public enum TransformDirection { RecoverIdeal = 0, DocumentForward = 1 }

    public sealed class PitchOptions
    {
        public bool Enabled { get; set; }
        public double AngleDegrees { get; set; }
        public double ScanPitchMm { get; set; }
        public double ZStepMm { get; set; }
        public double GrayOffset { get; set; }
        public double ZOriginMm { get; set; }
        public ScanAxis Axis { get; set; }
        public TransformDirection Direction { get; set; }
        public int ProjectionWorkers { get; set; }

        public PitchOptions()
        {
            Enabled = true;
            ZStepMm = 0.00105;
            GrayOffset = 32768.0;
            Axis = ScanAxis.Rows;
            Direction = TransformDirection.RecoverIdeal;
            ProjectionWorkers = 1;
        }
    }

    /// <summary>
    /// Packed row-major uint16 depth correction. Constructor snapshots options;
    /// reuse this instance across frames. No source pixels are modified.
    /// Immutable lookup tables are safe to share between concurrent calls when
    /// each call owns a different destination buffer.
    /// </summary>
    public sealed class PitchCorrector
    {
        private readonly bool enabled;
        private readonly ScanAxis axis;
        private readonly int workers;
        private readonly double[] shifts;
        private readonly ushort[] values;

        public PitchCorrector(PitchOptions options)
        {
            if (options == null) throw new ArgumentNullException("options");
            enabled = options.Enabled;
            if (!enabled) return;
            double angle = Finite(options.AngleDegrees, "AngleDegrees");
            double pitch = Finite(options.ScanPitchMm, "ScanPitchMm");
            double step = Finite(options.ZStepMm, "ZStepMm");
            double offset = Finite(options.GrayOffset, "GrayOffset");
            double origin = Finite(options.ZOriginMm, "ZOriginMm");
            axis = options.Axis;
            TransformDirection direction = options.Direction;
            workers = options.ProjectionWorkers;
            if (Math.Abs(angle) >= 60) throw new ArgumentOutOfRangeException("AngleDegrees", "Absolute angle must be less than 60 degrees.");
            if (pitch <= 0) throw new ArgumentOutOfRangeException("ScanPitchMm");
            if (step <= 0) throw new ArgumentOutOfRangeException("ZStepMm");
            if (workers < 1) throw new ArgumentOutOfRangeException("ProjectionWorkers");
            if (axis != ScanAxis.Rows && axis != ScanAxis.Columns) throw new ArgumentOutOfRangeException("Axis");
            if (direction != TransformDirection.RecoverIdeal && direction != TransformDirection.DocumentForward)
                throw new ArgumentOutOfRangeException("Direction");

            double radians = angle * (Math.PI / 180.0);
            double cosine = Math.Cos(radians), sine = Math.Sin(radians), tangent = Math.Tan(radians);
            shifts = new double[65536];
            values = new ushort[65536];
            for (int d = 1; d < 65535; ++d)
            {
                double z = (d - offset) * step + origin;
                double shift, correctedZ;
                if (direction == TransformDirection.RecoverIdeal)
                {
                    shift = -z * tangent;
                    correctedZ = z / cosine;
                }
                else
                {
                    shift = z * sine;
                    correctedZ = z * cosine;
                }
                // Keep the same operation order as the Python algorithm.
                // Do not round the shift separately from the pixel coordinate.
                double raw = Math.Round((correctedZ - origin) / step + offset, MidpointRounding.ToEven);
                double delta = shift / pitch;
                if (raw >= 1 && raw <= 65534 && !double.IsNaN(delta) && !double.IsInfinity(delta))
                {
                    values[d] = (ushort)raw;
                    shifts[d] = delta;
                }
            }
        }

        public ushort[] Correct(ushort[] input, int width, int height)
        {
            int length = CheckInput(input, width, height);
            ushort[] output = new ushort[length];
            CorrectInto(input, output, width, height);
            return output;
        }

        /// <summary>Output must be a distinct packed array of width*height values.</summary>
        public void CorrectInto(ushort[] input, ushort[] output, int width, int height)
        {
            int length = CheckInput(input, width, height);
            if (output == null) throw new ArgumentNullException("output");
            if (output.Length != length) throw new ArgumentException("Output length must equal width * height.", "output");
            if (Object.ReferenceEquals(input, output)) throw new ArgumentException("Input and output must be different arrays.", "output");
            if (!enabled)
            {
                Array.Copy(input, output, length);
                return;
            }
            Array.Clear(output, 0, length);
            // Each strip owns the entire scan axis: even large shifts and
            // collisions cannot cross into another worker's destination pixels.
            int extent = axis == ScanAxis.Rows ? width : height;
            int count = Math.Min(workers, extent);
            Action<int> project = delegate(int index)
            {
                int begin = (int)((long)extent * index / count);
                int end = (int)((long)extent * (index + 1) / count);
                if (axis == ScanAxis.Rows)
                {
                    for (int row = 0; row < height; ++row)
                        for (int col = begin; col < end; ++col)
                        {
                            ushort source = input[row * width + col];
                            ushort value = values[source];
                            if (value == 0) continue;
                            double mapped = Math.Round(row + shifts[source], MidpointRounding.ToEven);
                            if (!(mapped >= 0 && mapped < height)) continue;
                            int target = (int)mapped * width + col;
                            if (value > output[target]) output[target] = value;
                        }
                }
                else
                {
                    for (int row = begin; row < end; ++row)
                        for (int col = 0; col < width; ++col)
                        {
                            ushort source = input[row * width + col];
                            ushort value = values[source];
                            if (value == 0) continue;
                            double mapped = Math.Round(col + shifts[source], MidpointRounding.ToEven);
                            if (!(mapped >= 0 && mapped < width)) continue;
                            int target = row * width + (int)mapped;
                            if (value > output[target]) output[target] = value;
                        }
                }
            };
            if (count == 1) project(0);
            else Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = count }, project);
        }

        private static int CheckInput(ushort[] input, int width, int height)
        {
            if (input == null) throw new ArgumentNullException("input");
            if (width <= 0 || height <= 0 || (long)width * height > Int32.MaxValue)
                throw new ArgumentOutOfRangeException("width", "Width and height must define a valid array size.");
            int length = width * height;
            if (input.Length != length) throw new ArgumentException("Input length must equal width * height.", "input");
            return length;
        }

        private static double Finite(double value, string name)
        {
            if (double.IsNaN(value) || double.IsInfinity(value)) throw new ArgumentOutOfRangeException(name, "Value must be finite.");
            return value;
        }
    }
}

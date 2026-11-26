using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Reflection;
using System.Threading;

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

    public sealed class DepthPieceRange
    {
        public int PieceId { get; private set; }
        public int StartRow { get; private set; }
        public int EndRow { get; private set; }

        public DepthPieceRange(int pieceId, int startRow, int endRow)
        {
            if (pieceId <= 0) throw new ArgumentOutOfRangeException("pieceId");
            PieceId = pieceId;
            StartRow = startRow;
            EndRow = endRow;
        }
    }

    public sealed class CorrectedDepthPiece
    {
        public int PieceId { get; private set; }
        public ushort[] Depth { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int StartRow { get; private set; }
        public Dictionary<string, object> PitchCorrection { get; private set; }

        internal CorrectedDepthPiece(ushort[] depth, int width, int height, int startRow,
            Dictionary<string, object> pitchCorrection, int pieceId = 0)
        {
            PieceId = pieceId;
            Depth = depth;
            Width = width;
            Height = height;
            StartRow = startRow;
            PitchCorrection = pitchCorrection;
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
        private readonly ParallelOptions parallelOptions;
        private readonly Dictionary<string, object> correctionMetadata;

        public PitchCorrector(PitchOptions options)
        {
            if (options == null) throw new ArgumentNullException("options");
            enabled = options.Enabled;
            if (!enabled)
            {
                correctionMetadata = new Dictionary<string, object> { { "applied", false }, { "reason", "disabled" } };
                return;
            }
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
            parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = workers };
            correctionMetadata = new Dictionary<string, object>
            {
                { "applied", true },
                { "model", "smartray_pitch" },
                { "angle_deg", angle },
                { "scan_pitch_mm", pitch },
                { "z_step_mm", step },
                { "gray_offset", offset },
                { "z_origin_mm", origin },
                { "scan_axis", axis == ScanAxis.Rows ? "rows" : "columns" },
                { "transform_direction", direction == TransformDirection.RecoverIdeal ? "recover_ideal" : "document_forward" }
            };

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
            // CLR already zero-initializes new arrays. Do not clear it again.
            Project(input, output, width, height, false);
            return output;
        }

        /// <summary>Correct the full-width raw rows [startRow, endRow) in piece-local coordinates.</summary>
        public CorrectedDepthPiece CorrectPiece(ushort[] input, int width, int height, int startRow, int endRow)
        {
            CheckInput(input, width, height);
            CheckRows(startRow, endRow, height);
            return ProjectPiece(input, width, startRow, endRow, 0);
        }

        /// <summary>Correct each configured raw row range and publish it immediately in input order.</summary>
        public void CorrectPieces(ushort[] input, int width, int height, IEnumerable<DepthPieceRange> pieces,
            Action<CorrectedDepthPiece> onPieceReady, CancellationToken cancellationToken = default(CancellationToken))
        {
            CheckInput(input, width, height);
            if (pieces == null) throw new ArgumentNullException("pieces");
            if (onPieceReady == null) throw new ArgumentNullException("onPieceReady");
            var ranges = new List<DepthPieceRange>();
            var ids = new HashSet<int>();
            foreach (DepthPieceRange piece in pieces)
            {
                if (piece == null || !ids.Add(piece.PieceId))
                    throw new ArgumentException("Every piece must have a unique ID.", "pieces");
                CheckRows(piece.StartRow, piece.EndRow, height);
                ranges.Add(piece);
            }
            foreach (DepthPieceRange piece in ranges)
            {
                cancellationToken.ThrowIfCancellationRequested();
                onPieceReady(ProjectPiece(input, width, piece.StartRow, piece.EndRow, piece.PieceId));
            }
        }

        private static void CheckRows(int startRow, int endRow, int height)
        {
            if (startRow < 0 || startRow >= height) throw new ArgumentOutOfRangeException("startRow");
            if (endRow <= startRow || endRow > height) throw new ArgumentOutOfRangeException("endRow");
        }

        private CorrectedDepthPiece ProjectPiece(ushort[] input, int width, int startRow, int endRow, int pieceId)
        {
            int pieceHeight = endRow - startRow;
            var corrected = new ushort[width * pieceHeight];
            Project(input, corrected, width, pieceHeight, false, startRow * width);
            return new CorrectedDepthPiece(corrected, width, pieceHeight, startRow,
                new Dictionary<string, object>(correctionMetadata), pieceId);
        }

        /// <summary>Output must be a distinct packed array of width*height values.</summary>
        public void CorrectInto(ushort[] input, ushort[] output, int width, int height)
        {
            int length = CheckInput(input, width, height);
            if (output == null) throw new ArgumentNullException("output");
            if (output.Length != length) throw new ArgumentException("Output length must equal width * height.", "output");
            if (Object.ReferenceEquals(input, output)) throw new ArgumentException("Input and output must be different arrays.", "output");
            Project(input, output, width, height, true);
        }

        private void Project(ushort[] input, ushort[] output, int width, int height, bool clearOutput,
            int sourceOffset = 0)
        {
            int length = output.Length;
            if (!enabled)
            {
                Array.Copy(input, sourceOffset, output, 0, length);
                return;
            }
            // Each strip owns the entire scan axis: even large shifts and
            // collisions cannot cross into another worker's destination pixels.
            int extent = axis == ScanAxis.Rows ? width : height;
            int count = Math.Min(workers, extent);
            if (clearOutput && count == 1)
            {
                // One worker can use the contiguous bulk clear directly.
                Array.Clear(output, 0, length);
                clearOutput = false;
            }
            Action<int> project = delegate(int index)
            {
                int begin = (int)((long)extent * index / count);
                int end = (int)((long)extent * (index + 1) / count);
                if (axis == ScanAxis.Rows)
                {
                    if (clearOutput)
                    {
                        // Clear every row in our columns before projecting any
                        // source row: large shifts may target later rows.
                        for (int rowBase = 0; rowBase < length; rowBase += width)
                            Array.Clear(output, rowBase + begin, end - begin);
                    }
                    int sourceRow = sourceOffset;
                    for (int row = 0; row < height; ++row)
                    {
                        int sourceIndex = sourceRow + begin;
                        for (int col = begin; col < end; ++col)
                        {
                            ushort source = input[sourceIndex++];
                            ushort value = values[source];
                            if (value == 0) continue;
                            double mapped = Math.Round(row + shifts[source], MidpointRounding.ToEven);
                            if (!(mapped >= 0 && mapped < height)) continue;
                            int target = (int)mapped * width + col;
                            if (value > output[target]) output[target] = value;
                        }
                        sourceRow += width;
                    }
                }
                else
                {
                    int targetRow = begin * width;
                    if (clearOutput) Array.Clear(output, targetRow, (end - begin) * width);
                    for (int row = begin; row < end; ++row)
                    {
                        int sourceIndex = sourceOffset + targetRow;
                        for (int col = 0; col < width; ++col)
                        {
                            ushort source = input[sourceIndex++];
                            ushort value = values[source];
                            if (value == 0) continue;
                            double mapped = Math.Round(col + shifts[source], MidpointRounding.ToEven);
                            if (!(mapped >= 0 && mapped < width)) continue;
                            int target = targetRow + (int)mapped;
                            if (value > output[target]) output[target] = value;
                        }
                        targetRow += width;
                    }
                }
            };
            if (count == 1) project(0);
            else Parallel.For(0, count, parallelOptions, project);
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

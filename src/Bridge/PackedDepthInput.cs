using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace Bridge;

/// <summary>Depth transport is packed little-endian uint16 without row headers or padding.</summary>
internal static class PackedDepthInput
{
    internal static int ByteCount(int width, int height)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException("3D image_width and image_height must be positive.");
        return checked(checked(width * height) * sizeof(ushort));
    }

    internal static void Validate(int size, int width, int height, string format)
    {
        if (format != "uint16" && format != "raw_uint16")
            throw new ArgumentException("3D input requires raw little-endian uint16, not encoded PNG.");
        if (size != ByteCount(width, height))
            throw new ArgumentException("3D data_size must equal image_width * image_height * 2.");
    }

    // Offline sample adapter only: decode in the caller, never in the algorithm.
    internal static byte[] ReadFile(string path, out int width, out int height)
    {
        using var stream = File.OpenRead(path);
        var decoder = new PngBitmapDecoder(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        var frame = decoder.Frames[0];
        if (frame.Format != PixelFormats.Gray16)
            throw new InvalidDataException("3D sample must be an original single-channel 16-bit depth PNG.");
        width = frame.PixelWidth;
        height = frame.PixelHeight;
        var pixels = new byte[ByteCount(width, height)];
        frame.CopyPixels(pixels, checked(width * sizeof(ushort)), 0);
        return pixels;
    }
}

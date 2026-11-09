using System.Text.Json;
using System.Text.Json.Serialization;

namespace Bridge;

// 负责初始化通信服务
public class InitRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "init";
}

// 负责初始化算法服务
public class InitAlgorithmRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "init_algorithm";

    [JsonPropertyName("stage")]
    public string Stage { get; set; } = string.Empty;

    [JsonPropertyName("model_path")]
    public Dictionary<string, string> ModelPath { get; set; } = new();

    [JsonPropertyName("config_path")]
    public string? ConfigPath { get; set; }

    [JsonPropertyName("config_text")]
    public string? ConfigText { get; set; }
}

public class Detect2DRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "detect";

    [JsonPropertyName("stage")]
    public string Stage { get; set; } = "2d";

    // Streaming mode identifiers. They are optional for the legacy single-image API.
    [JsonPropertyName("board_id")]
    public string BoardId { get; set; } = string.Empty;

    [JsonPropertyName("piece_id")]
    public int PieceId { get; set; }

    [JsonPropertyName("shm_name")]
    public string ShmName { get; set; } = string.Empty;

    [JsonPropertyName("image_size")]
    public int ImageSize { get; set; }

    [JsonPropertyName("image_format")]
    public string ImageFormat { get; set; } = "rgb8";

    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }

    [JsonPropertyName("image_channels")]
    public int ImageChannels { get; set; }

    [JsonPropertyName("image_offset")]
    public long ImageOffset { get; set; }

    [JsonPropertyName("image_stride")]
    public int ImageStride { get; set; }

    /// <summary>Optional EfficientAD threshold override: one number or per-model dictionary.</summary>
    [JsonPropertyName("threshold")]
    public object? Threshold { get; set; }
}

public class Detect3DRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "detect";

    [JsonPropertyName("stage")]
    public string Stage { get; set; } = "3d";

    // Streaming mode identifier. It is optional for the legacy 3D-only API.
    [JsonPropertyName("board_id")]
    public string BoardId { get; set; } = string.Empty;

    [JsonPropertyName("shm_name")]
    public string ShmName { get; set; } = string.Empty;

    [JsonPropertyName("data_size")]
    public int DataSize { get; set; }

    [JsonPropertyName("data_format")]
    public string DataFormat { get; set; } = "uint16";

    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }

}

public class BoardImage2DInput
{
    [JsonPropertyName("piece_id")]
    public int PieceId { get; set; }

    [JsonPropertyName("shm_name")]
    public string ShmName { get; set; } = string.Empty;

    [JsonPropertyName("image_size")]
    public int ImageSize { get; set; }

    [JsonPropertyName("image_format")]
    public string ImageFormat { get; set; } = "encoded";

    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }

    [JsonPropertyName("image_channels")]
    public int ImageChannels { get; set; } = 3;
}

public class BoardImage3DInput
{
    [JsonPropertyName("shm_name")]
    public string ShmName { get; set; } = string.Empty;

    [JsonPropertyName("data_size")]
    public int DataSize { get; set; }

    [JsonPropertyName("data_format")]
    public string DataFormat { get; set; } = "uint16";

    [JsonPropertyName("image_width")]
    public int ImageWidth { get; set; }

    [JsonPropertyName("image_height")]
    public int ImageHeight { get; set; }
}

public class DetectBoardRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "detect_board";

    [JsonPropertyName("stage")]
    public string Stage { get; set; } = "board";

    [JsonPropertyName("batch_id")]
    public string BatchId { get; set; } = string.Empty;

    [JsonPropertyName("images_2d")]
    public List<BoardImage2DInput> Images2D { get; set; } = new();

    [JsonPropertyName("image_3d")]
    public BoardImage3DInput Image3D { get; set; } = new();
}

public class ReleaseRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "release";
}

public class ReleaseBoardRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "release_board";

    [JsonPropertyName("board_id")]
    public string BoardId { get; set; } = string.Empty;
}

public class ReloadDefectConfigRequest
{
    [JsonPropertyName("cmd")]
    public string Cmd { get; set; } = "reload_defect_config";

    [JsonPropertyName("config_path")]
    public string? ConfigPath { get; set; }

    [JsonPropertyName("config_text")]
    public string ConfigText { get; set; } = string.Empty;
}

public class BridgeBaseResponse
{
    [JsonPropertyName("request_id")]
    public string? RequestId { get; set; }
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("cmd")]
    public string? Cmd { get; set; }

    [JsonPropertyName("stage")]
    public string? Stage { get; set; }

    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("timings")]
    public Dictionary<string, double> Timings { get; set; } = new();

    [JsonPropertyName("model_load_timings")]
    public Dictionary<string, double> ModelLoadTimings { get; set; } = new();

    [JsonPropertyName("csharp_roundtrip_ms")]
    public double CsharpRoundtripMs { get; set; }

    [JsonPropertyName("csharp_total_ms")]
    public double CsharpTotalMs { get; set; }
}

public class DetectResponse : BridgeBaseResponse
{
    [JsonPropertyName("elapsed_ms")]
    public double ElapsedMs { get; set; }

    [JsonPropertyName("results_2d")]
    public Results2D? Results2D { get; set; }

    [JsonPropertyName("results_3d")]
    public Results3D? Results3D { get; set; }

    [JsonPropertyName("batch_id")]
    public string? BatchId { get; set; }

    [JsonPropertyName("board_id")]
    public string? BoardId { get; set; }

    [JsonPropertyName("piece_id")]
    public int? PieceId { get; set; }

    [JsonPropertyName("cache_state")]
    public CacheState? CacheState { get; set; }

    [JsonPropertyName("released")]
    public ReleaseSummary? Released { get; set; }

    [JsonPropertyName("results_board")]
    public ResultsBoard? ResultsBoard { get; set; }

    [JsonPropertyName("csharp_image_read_ms")]
    public double CsharpImageReadMs { get; set; }

    [JsonPropertyName("csharp_shared_memory_write_ms")]
    public double CsharpSharedMemoryWriteMs { get; set; }

    [JsonPropertyName("missing_piece_ids")]
    public List<int> MissingPieceIds { get; set; } = new();

    public List<RectangleResult2D> GetRectangles2D()
    {
        return Results2D?.Rectangles ?? new List<RectangleResult2D>();
    }

    public List<Piece3D> GetPieces3D()
    {
        return Results3D?.Pieces ?? new List<Piece3D>();
    }
}

public class ResultsBoard
{
    [JsonPropertyName("pitch_correction")]
    public Dictionary<string, JsonElement>? PitchCorrection { get; set; }

    [JsonPropertyName("depth_coordinate_space")]
    public string? DepthCoordinateSpace { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("registration_ok")]
    public bool RegistrationOk { get; set; }

    [JsonPropertyName("pieces")]
    public List<BoardPieceResult> Pieces { get; set; } = new();
}

public class BoardPieceResult
{
    [JsonPropertyName("pitch_correction")]
    public Dictionary<string, JsonElement>? PitchCorrection { get; set; }

    [JsonPropertyName("depth_coordinate_space")]
    public string? DepthCoordinateSpace { get; set; }

    [JsonPropertyName("piece_id")]
    public int PieceId { get; set; }
    [JsonPropertyName("timings")]
    public Dictionary<string, double> Timings { get; set; } = new();

    [JsonPropertyName("image_2d_size")]
    public List<int> Image2DSize { get; set; } = new();

    [JsonPropertyName("crop_2d")]
    public List<int> Crop2D { get; set; } = new();

    [JsonPropertyName("wire_rois_2d")]
    public List<List<int>> WireRois2D { get; set; } = new();

    [JsonPropertyName("wire_roi_checks")]
    public List<WireRoiCheck2D> WireRoiChecks { get; set; } = new();

    [JsonPropertyName("depth_start_row")]
    public int DepthStartRow { get; set; }

    [JsonPropertyName("depth_end_row")]
    public int DepthEndRow { get; set; }

    /// <summary>"board_3d" means height points refer to the unsliced full depth image.</summary>
    [JsonPropertyName("height_point_coordinate_space")]
    public string? HeightPointCoordinateSpace { get; set; }

    [JsonPropertyName("registration")]
    public RegistrationResult Registration { get; set; } = new();

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("reference_planes")]
    public ReferencePlanesResult ReferencePlanes { get; set; } = new();

    [JsonPropertyName("wire_measurements")]
    public List<WireMeasurementResult> WireMeasurements { get; set; } = new();

    [JsonPropertyName("solder_measurements")]
    public List<SolderMeasurementResult> SolderMeasurements { get; set; } = new();

    [JsonPropertyName("chip_anomaly_2d")]
    public ChipAnomaly2DResult? ChipAnomaly2D { get; set; }

    [JsonPropertyName("silver_overflow_2d")]
    public Diagnostic2D? SilverOverflow2D { get; set; }

    [JsonPropertyName("silver_continuity_2d")]
    public Diagnostic2D? SilverContinuity2D { get; set; }

    [JsonPropertyName("masks")]
    public List<AlignedMaskResult> Masks { get; set; } = new();
}

public class RegistrationResult
{
    [JsonPropertyName("scale")]
    public double Scale { get; set; }

    [JsonPropertyName("tx")]
    public double Tx { get; set; }

    [JsonPropertyName("ty")]
    public double Ty { get; set; }

    [JsonPropertyName("score")]
    public double Score { get; set; }

    [JsonPropertyName("is_valid")]
    public bool IsValid { get; set; }

    [JsonPropertyName("matrix_2d_to_3d")]
    public List<List<double>> Matrix2DTo3D { get; set; } = new();
}

public class AlignedMaskResult
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_id")]
    public int ClassId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("mask_2d")]
    public MaskGeometry Mask2D { get; set; } = new();

    [JsonPropertyName("mask_3d_local")]
    public MaskGeometry Mask3DLocal { get; set; } = new();

    [JsonPropertyName("bending_2d")]
    public WireBendingResult? Bending2D { get; set; }

    [JsonPropertyName("height_3d")]
    public HeightMeasurementResult? Height3D { get; set; }

    [JsonPropertyName("bond_location")]
    public string? BondLocation { get; set; }

    [JsonPropertyName("reference_instance_id")]
    public int? ReferenceInstanceId { get; set; }

    [JsonPropertyName("chip_inside")]
    public bool? ChipInside { get; set; }

    [JsonPropertyName("plane_3d")]
    public PlaneResult? Plane3D { get; set; }

    [JsonPropertyName("chip_anomaly_2d")]
    public ChipAnomalyEntry2D? ChipAnomaly2D { get; set; }
}

public class WireBendingResult
{
    [JsonPropertyName("is_valid")]
    public bool IsValid { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("mask_ratio")]
    public double MaskRatio { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("roi_id")]
    public int? RoiId { get; set; }

    [JsonPropertyName("roi_mask_pixel_count")]
    public int RoiMaskPixelCount { get; set; }

    [JsonPropertyName("covered_mask_pixel_count")]
    public int CoveredMaskPixelCount { get; set; }

    // Four edge intersections, ordered around the evaluation quadrilateral.
    [JsonPropertyName("rectangle")]
    public List<List<int>> Rectangle { get; set; } = new();
}

public class WireMeasurementResult
{
    [JsonPropertyName("region_id")]
    public int RegionId { get; set; }

    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("mask_2d")]
    public MaskGeometry Mask2D { get; set; } = new();

    [JsonPropertyName("mask_3d_local")]
    public MaskGeometry Mask3DLocal { get; set; } = new();

    [JsonPropertyName("bending_2d")]
    public WireBendingResult Bending2D { get; set; } = new();

    [JsonPropertyName("height_3d")]
    public HeightMeasurementResult Height3D { get; set; } = new();
}

public class SolderMeasurementResult
{
    [JsonPropertyName("measurement_id")]
    public int MeasurementId { get; set; }

    [JsonPropertyName("source_instance_id")]
    public int SourceInstanceId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("endpoint")]
    public string Endpoint { get; set; } = string.Empty;

    [JsonPropertyName("location")]
    public string Location { get; set; } = string.Empty;

    [JsonPropertyName("reference_instance_id")]
    public int? ReferenceInstanceId { get; set; }

    [JsonPropertyName("chip_inside")]
    public bool ChipInside { get; set; }

    [JsonPropertyName("reference_plane")]
    public PlaneResult? ReferencePlane { get; set; }

    [JsonPropertyName("mask_2d")]
    public MaskGeometry Mask2D { get; set; } = new();

    [JsonPropertyName("mask_3d_local")]
    public MaskGeometry Mask3DLocal { get; set; } = new();

    [JsonPropertyName("height_3d")]
    public HeightMeasurementResult Height3D { get; set; } = new();
}

public class HeightMeasurementResult
{
    [JsonPropertyName("reference")]
    public string Reference { get; set; } = string.Empty;

    [JsonPropertyName("is_valid")]
    public bool IsValid { get; set; }

    [JsonPropertyName("is_ok")]
    public bool? IsOk { get; set; }

    [JsonPropertyName("height_mm")]
    public double? HeightMm { get; set; }

    [JsonPropertyName("height_diff_mm")]
    public double? HeightDiffMm { get; set; }

    [JsonPropertyName("point")]
    public ImagePoint? Point { get; set; }

    /// <summary>Joint detections return "board_3d" or "pitch_corrected_board_3d"; older responses may omit this field.</summary>
    [JsonPropertyName("point_coordinate_space")]
    public string? PointCoordinateSpace { get; set; }
}

public class ImagePoint
{
    [JsonPropertyName("x")]
    public int X { get; set; }

    [JsonPropertyName("y")]
    public int Y { get; set; }
}

public class ReferencePlanesResult
{
    [JsonPropertyName("wire_platform")]
    public PlaneResult? WirePlatform { get; set; }

    [JsonPropertyName("chips")]
    public List<ChipPlaneResult> Chips { get; set; } = new();
}

public class ChipPlaneResult
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("plane")]
    public PlaneResult? Plane { get; set; }
}

public class PlaneResult
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("coefficients")]
    public List<double> Coefficients { get; set; } = new();

    [JsonPropertyName("valid_points")]
    public int ValidPoints { get; set; }
}

public class MaskGeometry
{
    [JsonPropertyName("area_px")]
    public int AreaPx { get; set; }

    [JsonPropertyName("bounding_box")]
    public List<int> BoundingBox { get; set; } = new();

    [JsonPropertyName("contours")]
    public List<List<List<int>>> Contours { get; set; } = new();
}

public class Results2D
{
    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("rectangles")]
    public List<RectangleResult2D> Rectangles { get; set; } = new();

    [JsonPropertyName("circle")]
    public CircleResult2D? Circle { get; set; }

    [JsonPropertyName("crop_box")]
    public List<int> CropBox { get; set; } = new();

    [JsonPropertyName("wire_rois_2d")]
    public List<List<int>> WireRois2D { get; set; } = new();

    [JsonPropertyName("instances")]
    public List<InstanceResult2D> Instances { get; set; } = new();

    [JsonPropertyName("wire_bending_2d")]
    public WireBending2DResult? WireBending2D { get; set; }

    [JsonPropertyName("silver_continuity_2d")]
    public Diagnostic2D? SilverContinuity2D { get; set; }

    [JsonPropertyName("silver_overflow_2d")]
    public Diagnostic2D? SilverOverflow { get; set; }

    [JsonPropertyName("chip_rotation_2d")]
    public ChipAlignment2DResult? ChipRotation2D { get; set; }

    [JsonPropertyName("chip_anomaly_2d")]
    public ChipAnomaly2DResult? ChipAnomaly2D { get; set; }
}

public class InstanceResult2D
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_id")]
    public int ClassId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("mask_2d")]
    public MaskGeometry Mask2D { get; set; } = new();

    [JsonPropertyName("rectangle")]
    public List<List<int>> Rectangle { get; set; } = new();

    [JsonPropertyName("rectangle_ratio")]
    public double RectangleRatio { get; set; }

    [JsonPropertyName("bending_2d")]
    public WireBendingResult? Bending2D { get; set; }

    [JsonPropertyName("chip_rotation_2d")]
    public ChipAlignmentEntry2D? ChipRotation2D { get; set; }

    [JsonPropertyName("chip_anomaly_2d")]
    public ChipAnomalyEntry2D? ChipAnomaly2D { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }
}

public class CircleResult2D
{
    [JsonPropertyName("x")]
    public int X { get; set; }

    [JsonPropertyName("y")]
    public int Y { get; set; }

    [JsonPropertyName("radius")]
    public int Radius { get; set; }
}

public class WireMeasurement2DResult
{
    [JsonPropertyName("region_id")]
    public int RegionId { get; set; }

    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("mask_2d")]
    public MaskGeometry Mask2D { get; set; } = new();

    [JsonPropertyName("bending_2d")]
    public WireBendingResult Bending2D { get; set; } = new();
}

public class WireRoiCheck2D
{
    [JsonPropertyName("roi_id")]
    public int RoiId { get; set; }

    [JsonPropertyName("box")]
    public List<int> Box { get; set; } = new();

    [JsonPropertyName("mask_pixel_count")]
    public int MaskPixelCount { get; set; }

    [JsonPropertyName("instance_ids")]
    public List<int> InstanceIds { get; set; } = new();

    [JsonPropertyName("bending_checks")]
    public List<WireRoiBendingCheck2D> BendingChecks { get; set; } = new();

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
}

public class WireBending2DResult
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("is_ok")]
    public bool? IsOk { get; set; }

    [JsonPropertyName("roi_checks")]
    public List<WireRoiCheck2D> RoiChecks { get; set; } = new();

    [JsonPropertyName("measurements")]
    public List<WireMeasurement2DResult> Measurements { get; set; } = new();
}

public class WireRoiBendingCheck2D
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("mask_ratio")]
    public double MaskRatio { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "";

    [JsonPropertyName("is_valid")]
    public bool IsValid { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }
}

public class ChipAlignment2DResult
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("is_ok")]
    public bool? IsOk { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("max_angle_deg")]
    public double MaxAngleDeg { get; set; }

    [JsonPropertyName("chips")]
    public List<ChipAlignmentEntry2D> Chips { get; set; } = new();
}

public class ChipAlignmentEntry2D
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("center_xy")]
    public List<double> CenterXy { get; set; } = new();

    [JsonPropertyName("rectangle")]
    public List<List<int>> Rectangle { get; set; } = new();

    [JsonPropertyName("width_px")]
    public double WidthPx { get; set; }

    [JsonPropertyName("height_px")]
    public double HeightPx { get; set; }

    [JsonPropertyName("angle_deg")]
    public double AngleDeg { get; set; }

    [JsonPropertyName("axis_deviation_deg")]
    public double AxisDeviationDeg { get; set; }

    [JsonPropertyName("max_angle_deg")]
    public double MaxAngleDeg { get; set; }

    [JsonPropertyName("mask_fill_ratio")]
    public double MaskFillRatio { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
}

public class ChipAnomaly2DResult
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("is_ok")]
    public bool? IsOk { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("ratio_definition")]
    public string RatioDefinition { get; set; } = string.Empty;

    [JsonPropertyName("aspect_ratio_models")]
    public List<ChipAspectRatioModel2D> AspectRatioModels { get; set; } = new();

    [JsonPropertyName("chips")]
    public List<ChipAnomalyEntry2D> Chips { get; set; } = new();
}

public class ChipAnomalyEntry2D
{
    [JsonPropertyName("instance_id")]
    public int InstanceId { get; set; }

    [JsonPropertyName("class_name")]
    public string ClassName { get; set; } = string.Empty;

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;

    [JsonPropertyName("category")]
    public string Category { get; set; } = string.Empty;

    [JsonPropertyName("aspect_ratio")]
    public double AspectRatio { get; set; }

    [JsonPropertyName("model_aspect_ratio_range")]
    public List<double> ModelAspectRatioRange { get; set; } = new();

    [JsonPropertyName("rectification_angle_deg")]
    public double RectificationAngleDeg { get; set; }

    [JsonPropertyName("rectangle")]
    public List<List<int>> Rectangle { get; set; } = new();

    [JsonPropertyName("rectified_size")]
    public List<int> RectifiedSize { get; set; } = new();

    [JsonPropertyName("model_input_size")]
    public List<int> ModelInputSize { get; set; } = new();

    [JsonPropertyName("score")]
    public double Score { get; set; }

    [JsonPropertyName("score_topk")]
    public double ScoreTopK { get; set; }

    [JsonPropertyName("score_max")]
    public double ScoreMax { get; set; }

    [JsonPropertyName("threshold")]
    public double Threshold { get; set; }

    [JsonPropertyName("prediction")]
    public string Prediction { get; set; } = string.Empty;

    [JsonPropertyName("decision_rule")]
    public string DecisionRule { get; set; } = string.Empty;

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("active_scales")]
    public List<int> ActiveScales { get; set; } = new();

    [JsonPropertyName("anomaly_boxes")]
    public List<ChipAnomalyBox2D> AnomalyBoxes { get; set; } = new();
}

public class ChipAspectRatioModel2D
{
    [JsonPropertyName("range")]
    public List<double> Range { get; set; } = new();

    [JsonPropertyName("model")]
    public string Model { get; set; } = string.Empty;
}

public class ChipAnomalyBox2D
{
    [JsonPropertyName("crop_xyxy")]
    public List<int> CropXyxy { get; set; } = new();

    [JsonPropertyName("image_polygon")]
    public List<List<int>> ImagePolygon { get; set; } = new();
}

public class Diagnostic2D
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("is_ok")]
    public bool? IsOk { get; set; }

    [JsonPropertyName("message")]
    public string? Message { get; set; }

    [JsonPropertyName("boundary")]
    public BoundaryResult2D? Boundary { get; set; }

    [JsonPropertyName("measurement")]
    public System.Text.Json.JsonElement? Measurement { get; set; }

    [JsonPropertyName("registration")]
    public BoundaryRegistration2D? Registration { get; set; }
}

public class BoundaryResult2D
{
    [JsonPropertyName("mode")]
    public string Mode { get; set; } = string.Empty;

    [JsonPropertyName("points")]
    public List<List<double>> Points { get; set; } = new();

    [JsonPropertyName("coordinate_system")]
    public string CoordinateSystem { get; set; } = string.Empty;
    [JsonPropertyName("is_within_crop")]
    public bool? IsWithinCrop { get; set; }
}

public class BoundaryRegistration2D
{
    [JsonPropertyName("method")]
    public string Method { get; set; } = string.Empty;
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;
    [JsonPropertyName("is_valid")]
    public bool IsValid { get; set; }
    [JsonPropertyName("fallback_used")]
    public bool FallbackUsed { get; set; }
    [JsonPropertyName("matches")]
    public int? Matches { get; set; }
    [JsonPropertyName("inliers")]
    public int? Inliers { get; set; }
    [JsonPropertyName("inlier_ratio")]
    public double? InlierRatio { get; set; }
    [JsonPropertyName("reprojection_error_px")]
    public double? ReprojectionErrorPx { get; set; }
    [JsonPropertyName("template_name")]
    public string? TemplateName { get; set; }
    [JsonPropertyName("template_size")]
    public List<int> TemplateSize { get; set; } = new();
    [JsonPropertyName("template_sha256")]
    public string? TemplateSha256 { get; set; }
    [JsonPropertyName("message")]
    public string? Message { get; set; }
    [JsonPropertyName("homography_template_to_image")]
    public List<List<double>> HomographyTemplateToImage { get; set; } = new();
}

public class CacheState
{
    [JsonPropertyName("board_id")]
    public string BoardId { get; set; } = string.Empty;

    [JsonPropertyName("expected_piece_count")]
    public int ExpectedPieceCount { get; set; }

    [JsonPropertyName("received_piece_count")]
    public int ReceivedPieceCount { get; set; }

    [JsonPropertyName("piece_ids")]
    public List<int> PieceIds { get; set; } = new();

    [JsonPropertyName("ready_for_3d")]
    public bool ReadyFor3D { get; set; }
}

public class ReleaseSummary
{
    [JsonPropertyName("board_id")]
    public string BoardId { get; set; } = string.Empty;

    [JsonPropertyName("released_2d_count")]
    public int Released2DCount { get; set; }

    [JsonPropertyName("released_3d")]
    public bool Released3D { get; set; }
}

public class RectangleResult2D
{
    [JsonPropertyName("points")]
    public List<List<int>> Points { get; set; } = new();

    [JsonPropertyName("mask_ratio")]
    public double MaskRatio { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }
}

public class Results3D
{
    [JsonPropertyName("pitch_correction")]
    public Dictionary<string, JsonElement>? PitchCorrection { get; set; }

    [JsonPropertyName("depth_coordinate_space")]
    public string? DepthCoordinateSpace { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("pieces")]
    public List<Piece3D> Pieces { get; set; } = new();
}

public class Piece3D
{
    [JsonPropertyName("piece_id")]
    public int PieceId { get; set; }

    [JsonPropertyName("is_ok")]
    public bool IsOk { get; set; }

    [JsonPropertyName("substrate_height_mm")]
    public double SubstrateHeightMm { get; set; }

    [JsonPropertyName("points")]
    public List<DepthPoint3D> Points { get; set; } = new();
}

public class DepthPoint3D
{
    [JsonPropertyName("height_mm")]
    public double HeightMm { get; set; }

    [JsonPropertyName("height_diff_mm")]
    public double HeightDiffMm { get; set; }

    [JsonPropertyName("x")]
    public int X { get; set; }

    [JsonPropertyName("y")]
    public int Y { get; set; }
}

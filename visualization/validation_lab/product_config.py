"""Load the product configuration selected for a validation session."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from algorithm.runtime.config import feature_parameters, load_runtime_config
from algorithm.vision3d.depth.depth_preprocessing import preprocessing_parameters
from algorithm.vision3d.depth.segment import SegmentationConfig
from detect_and_view import algorithm_request


DEFAULT_PRODUCT = "normal"


def product_manifests(config_dir: Path) -> dict[str, Path]:
    return {path.parent.name: path for path in sorted((config_dir / "products").glob("*/config.json"))
            if path.is_file()}


@dataclass(frozen=True)
class ProductConfig:
    id: str
    version: str
    request: dict
    piece_count: int
    pitch: dict

    @property
    def metadata(self):
        return dict(id=self.id, label=self.id, version=self.version)


def load_product(config_dir: Path, product_id: str) -> ProductConfig:
    manifests = product_manifests(config_dir)
    if not isinstance(product_id, str) or product_id not in manifests:
        raise ValueError("找不到该产品型号的配置")
    request = algorithm_request(None, None, config_path=manifests[product_id])
    config = load_runtime_config(request["config_text"], request["config_path"],
                                 config_3d_path=request["config_3d_path"])
    piece_count = int(feature_parameters(config, "common_3d")["input"]["piece_count"])
    segmentation = SegmentationConfig.from_dict(feature_parameters(config, "segmentation_3d"))
    if piece_count <= 0 or segmentation.expected_piece_count != piece_count:
        raise ValueError("型号配置的片数与分片设置不一致")
    pitch = preprocessing_parameters(config)
    calibration = config["dynamic_affine_3d"]["parameters"].get("calibration")
    if isinstance(calibration, dict) and "profile" in calibration:
        calibration.pop("profile_path", None)
    request["config_text"] = json.dumps(config, sort_keys=True, ensure_ascii=True)
    version = hashlib.sha256(request["config_text"].encode("utf-8")).hexdigest()
    return ProductConfig(product_id, version, request, piece_count, pitch)

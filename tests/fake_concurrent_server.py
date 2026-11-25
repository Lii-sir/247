"""Protocol fixture used by the C# integration tests; not a production algorithm."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector
from transport.depth_input import validate_depth_input, depth_from_uint16
from transport.shared_memory import read_shared_memory
from algorithm.runtime.config import load_runtime_config


class RawDepthDetector(FakeDetector):
    def prepare_2d_piece(self, req):
        if req["board_id"] in {"release-pending", "never-complete"}:
            time.sleep(.8)
        return super().prepare_2d_piece(req)

    def _read_3d(self, req):
        self.depth_started.set()
        name, size, width, height = validate_depth_input(req)
        depth = depth_from_uint16(read_shared_memory(name, size), width, height)
        if depth.shape != (10, 3) or depth.ravel()[:4].tolist() != [1, 256, 32768, 65535]:
            raise ValueError("C# synthetic raw depth values or dimensions changed in transit")
        return depth

    def measure_registered_piece(self, piece_id, entry, depth, piece, registration, pitch_correction=None):
        result = super().measure_registered_piece(piece_id, entry, depth, piece, registration, pitch_correction)
        if "depth_coordinate_space" in piece:
            result.update(depth_coordinate_space=piece["depth_coordinate_space"],
                          depth_start_row=piece["y1"], depth_end_row=piece["y2"],
                          pitch_correction=pitch_correction)
        return result

class FixtureServer(AlgorithmServer):
    def handle_init_algorithm(self, req):
        config = load_runtime_config(req["config_text"], req["config_path"])
        if not isinstance(config.get("inspection"), dict):
            return dict(ok=False, error="Independent inspection configuration was not sent")
        return dict(ok=True, cmd="init_algorithm")


server = FixtureServer()
server.board = RawDepthDetector(delay=0.015)
server.run()

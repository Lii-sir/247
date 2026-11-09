"""Protocol fixture used by the C# integration tests; not a production algorithm."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector
from transport.depth_input import validate_depth_input, depth_from_uint16
from transport.shared_memory import read_shared_memory


class RawDepthDetector(FakeDetector):
    def _read_3d(self, req):
        self.depth_started.set()
        name, size, width, height = validate_depth_input(req)
        depth = depth_from_uint16(read_shared_memory(name, size), width, height)
        if depth.shape != (100, 3) or depth.ravel()[:4].tolist() != [1, 256, 32768, 65535]:
            raise ValueError("C# synthetic raw depth values or dimensions changed in transit")
        return depth

server = AlgorithmServer()
server.board = RawDepthDetector(delay=0.015)
server.run()

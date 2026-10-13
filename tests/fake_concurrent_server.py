"""Protocol fixture used by the C# integration tests; not a production algorithm."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from transport.bridge_server import AlgorithmServer
from pipeline_fakes import FakeDetector

server = AlgorithmServer()
server.board = FakeDetector(delay=0.015)
server.run()

"""集中定义项目路径，避免源码移动后将 outputs/assets 写进包目录。"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = PROJECT_DIR / "outputs"
ASSET_DIR = PROJECT_DIR / "assets"

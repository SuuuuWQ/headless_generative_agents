"""
Local configuration for the headless Generative Agents copy.

The code in this folder is intended to be runnable without the visual frontend
process. By default it still points at the original repository's storage and
asset directories so we can reuse existing simulations and maps.
"""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
key_owner = os.environ.get("KEY_OWNER", "local")

maze_assets_loc = os.environ.get(
    "MAZE_ASSETS_LOC",
    str(ROOT / "environment" / "frontend_server" / "static_dirs" / "assets"),
)
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = os.environ.get(
    "FS_STORAGE",
    str(ROOT / "environment" / "frontend_server" / "storage"),
)
fs_temp_storage = os.environ.get(
    "FS_TEMP_STORAGE",
    str(ROOT / "environment" / "frontend_server" / "temp_storage"),
)

collision_block_id = os.environ.get("COLLISION_BLOCK_ID", "32125")
debug = os.environ.get("DEBUG", "False").lower() == "true"

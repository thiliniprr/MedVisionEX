"""Central path resolution so the repo is portable: everything is anchored to
the repo root and overridable by environment variables. No absolute paths are
hardcoded anywhere else.

Env overrides (all optional):
  HF_HOME         HuggingFace cache root      (default: <repo>/.hf_cache)
  MEDVQA_DATA     dataset root                (default: <repo>/data)
  MEDVQA_IMAGES   image directory             (default: <data>/Images)
  MEDVQA_OUTPUTS  checkpoints/indices/results (default: <repo>/outputs)
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _resolve(env_var: str, default: Path) -> Path:
    val = os.environ.get(env_var)
    return Path(val).expanduser() if val else default


# HuggingFace cache (models + datasets). Setting HF_HOME makes the HF libraries
# use this location; we set it here so a fresh clone caches inside the repo.
HF_HOME = _resolve("HF_HOME", REPO_ROOT / ".hf_cache")
os.environ.setdefault("HF_HOME", str(HF_HOME))
HF_HUB = HF_HOME / "hub"

DATASET_ROOT = _resolve("MEDVQA_DATA", REPO_ROOT / "data")
IMAGES_DIR = _resolve("MEDVQA_IMAGES", DATASET_ROOT / "Images")
OUTPUTS_ROOT = _resolve("MEDVQA_OUTPUTS", REPO_ROOT / "outputs")

for _p in (HF_HUB, DATASET_ROOT, OUTPUTS_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# String forms for configs that expect plain strings.
HF_HUB_STR = str(HF_HUB)
DATASET_ROOT_STR = str(DATASET_ROOT)
IMAGES_DIR_STR = str(IMAGES_DIR)
OUTPUTS_ROOT_STR = str(OUTPUTS_ROOT)

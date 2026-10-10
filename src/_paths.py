"""Centralised path resolution for the open-source release.

Resolution order for the dataset HDF5:
  1. Environment variable VALVE_DATA (absolute or relative path), if set.
  2. ../data/dataset_face.h5  (full dataset, if the user placed it there).
  3. ../data/dataset_demo.h5  (the small demo subset shipped with the repo).

This keeps every script working out-of-the-box on the demo subset, while
letting users point to the full dataset without editing source files:

    export VALVE_DATA=/path/to/dataset_face.h5     # Linux / macOS
    set    VALVE_DATA=C:\\path\\to\\dataset_face.h5  # Windows
"""
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(os.path.dirname(_SRC_DIR), "data")


def resolve_data_path():
    env = os.environ.get("VALVE_DATA")
    if env:
        return os.path.abspath(env)
    full = os.path.join(_DATA_DIR, "dataset_face.h5")
    if os.path.exists(full):
        return full
    return os.path.join(_DATA_DIR, "dataset_demo.h5")


def resolve_ckpt_dir():
    """Directory holding released checkpoints."""
    return os.path.join(os.path.dirname(_SRC_DIR), "checkpoints")


DATA_PATH = resolve_data_path()
CKPT_DIR = resolve_ckpt_dir()

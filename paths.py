# paths.py
from pathlib import Path

# Anchor everything to this file's location (project root)
ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "Images"
PICKLE_DIR = DATA_DIR / "PICKLES"
BBOX_DIR = DATA_DIR / "BBoxes"
TEACHER_DIR = DATA_DIR / "Teacher"
PATCH_DIR = DATA_DIR / "Patch_Data"
PATCH_DATASET_DIR = PATCH_DIR / "dataset"
DATASET_DIR = DATA_DIR / "dataset"

# Subdirectories for the COCO-style dataset
OUTPUT_DIR = DATASET_DIR / "output"
TEST_DIR = DATASET_DIR / "test"
TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

# Subdirectories for the COCO-style patch dataset
PATCH_OUTPUT_DIR = PATCH_DATASET_DIR / "output"
PATCH_TEST_DIR = PATCH_DATASET_DIR / "test"
PATCH_TRAIN_DIR = PATCH_DATASET_DIR / "train"
PATCH_VALID_DIR = PATCH_DATASET_DIR / "valid"

# Code directories
CODE_DIR = ROOT / "code"
AGENT_DIR = CODE_DIR / "agent"
COLLECT_DIR = CODE_DIR / "collection_and_labeling"
STUDENT_DIR = CODE_DIR / "student"
TEACHER_DIR = CODE_DIR / "teacher"

# Auto-create data dirs that need to exist
_dirs_to_create = [
    DATA_DIR, IMAGE_DIR, PICKLE_DIR, BBOX_DIR, TEACHER_DIR, PATCH_DIR, PATCH_DATASET_DIR, DATASET_DIR,
    OUTPUT_DIR, TEST_DIR, TRAIN_DIR, VALID_DIR,
    PATCH_OUTPUT_DIR, PATCH_TEST_DIR, PATCH_TRAIN_DIR, PATCH_VALID_DIR
    ]
for d in _dirs_to_create:
    d.mkdir(parents=True, exist_ok=True)
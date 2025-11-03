from pathlib import Path

# Define the absolute path to the project's root directory.
# `Path(__file__)` is this file (settings.py)
# `.resolve()` makes it an absolute path
# `.parent` goes up to configs/
# `.parent` goes up to cnn-video-analysis/ (the root)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Core Directories ---
SRC_DIR = BASE_DIR / "src"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
TESTS_DIR = BASE_DIR / "tests"
SCRIPTS_DIR = BASE_DIR / "scripts"
DOCS_DIR = BASE_DIR / "docs"

# --- Data Subdirectories ---
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MODEL_FILE = MODELS_DIR / "yolo8_nano_640_43750_otimizado.pt"

# --- Outputs Subdirectories ---
OUTPUTS_DIR = BASE_DIR / "outputs"
INFERENCE_RESULTS_DIR = OUTPUTS_DIR / "inference_results"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SELECTED_VIDEOS_DIR = OUTPUTS_DIR / "selected_videos"


def create_dirs():
    """Creates the essential directories if they don't already exist."""
    for dir_path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        INFERENCE_RESULTS_DIR,
        VISUALIZATIONS_DIR,
        REPORTS_DIR,
        SELECTED_VIDEOS_DIR,
        LOGS_DIR,
    ]:
        # `parents=True` creates parent folders if needed
        # `exist_ok=True` doesn't raise an error if the folder already exists
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print("Creating essential directories...")
    create_dirs()
    print("All directories created successfully.")
from pathlib import Path

# Repo root = parent of /src
REPO_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

REPORTS_DIR = REPO_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

RANDOM_SEED = 42

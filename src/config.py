"""
Central place for paths and a few project-wide constants.

Keeping these in one file means the rest of the code never has to guess
where data lives or hard-code a magic number twice.
"""

from pathlib import Path

# Resolve everything relative to the project root so the code runs the
# same whether you launch it from src/, the repo root, or inside Docker.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

for _d in (PROCESSED_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The raw Superstore files are semicolon-separated, and two of them were
# exported in latin-1 rather than utf-8. Encode that knowledge once here.
RAW_FILES = {
    "orders": ("Orders.csv", "utf-8"),
    "products": ("Products.csv", "latin-1"),
    "customers": ("Customers.csv", "latin-1"),
    "location": ("Location.csv", "utf-8"),
}

# We forecast on a weekly grid (Mondays). Daily is far too sparse for most
# subcategories; weekly gives continuous, learnable series.
WEEK_ANCHOR = "W-MON"

# Default holdout length used when evaluating models, in the unit of each
# series (e.g. 6 for "last 6 months" at the monthly grain).
DEFAULT_TEST_MONTHS = 6

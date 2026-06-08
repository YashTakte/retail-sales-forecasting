"""
Entry point for Hugging Face Spaces (and any "just run it" launch).

Spaces looks for an `app.py` at the repo root and runs it. On a fresh
Space the processed tables and trained models don't exist yet, so we build
them once on first boot, then launch the Gradio UI.

Locally you can still run `python app/gradio_app.py` directly if you've
already run the pipeline and training yourself.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "app"))

import config  # noqa: E402


def _ensure_ready():
    """Build processed data and train models if they're not present yet."""
    processed = config.PROCESSED_DIR / "monthly_subcategory.csv"
    model = config.MODEL_DIR / "lightgbm_subcategory_units.pkl"

    if not processed.exists():
        print("First boot: building processed data...")
        import data_pipeline
        data_pipeline.build_all()

    if not model.exists():
        print("First boot: training models (this takes a minute)...")
        import train
        train.run()


if __name__ == "__main__":
    _ensure_ready()
    import gradio_app
    # Spaces provides the port via the platform; default to 7860 locally.
    gradio_app.build_ui().launch(server_name="0.0.0.0", server_port=7860)

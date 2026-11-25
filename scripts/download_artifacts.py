"""Compatibility entry point for the dataset downloader."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "download_data.py"), run_name="__main__")

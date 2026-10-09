"""Download the MediaPipe Face Landmarker model into models/."""
from pathlib import Path
import sys
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / "face_landmarker.task"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already present: {dest}")
        return 0
    print(f"Downloading {MODEL_URL}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

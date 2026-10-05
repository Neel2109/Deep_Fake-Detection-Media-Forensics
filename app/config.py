from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
REPORT_DIR = BASE_DIR / "data" / "reports"
DATABASE_PATH = Path(
    os.getenv("DEEPTRACE_DATABASE_PATH", str(BASE_DIR / "data" / "deeptrace.sqlite3"))
).expanduser()
WEIGHTS_DIR = BASE_DIR / "weights"
XCEPTION_CHECKPOINT = Path(
    os.getenv("XCEPTION_CHECKPOINT", str(WEIGHTS_DIR / "xception_deepfake.pth"))
).expanduser()
ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"},
    "video": {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".aac"},
}

def ensure_storage_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


APP_TITLE = os.getenv("APP_TITLE", "DeepTrace AI")
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))

"""Safe Pillow image loading with EXIF orientation applied."""

from pathlib import Path

from PIL import Image, ImageOps


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
        return image

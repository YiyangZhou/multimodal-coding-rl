"""Bound judge-only image inputs to Quality judge's observed 30,000-patch limit."""

import io
import warnings

from PIL import Image, ImageOps

PATCH_SIZE = 32
MAX_PATCHES = 30000


def patch_count(width, height):
    return ((width + PATCH_SIZE - 1) // PATCH_SIZE) * ((height + PATCH_SIZE - 1) // PATCH_SIZE)


def fit_size(width, height):
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    if patch_count(width, height) <= MAX_PATCHES:
        return width, height
    low, high = 0.0, 1.0
    for _ in range(48):
        scale = (low + high) / 2
        size = max(1, int(width * scale)), max(1, int(height * scale))
        if patch_count(*size) <= MAX_PATCHES:
            low = scale
        else:
            high = scale
    return max(1, int(width * low)), max(1, int(height * low))


def bounded_image_bytes(data):
    """Preserve normal inputs byte-for-byte; resize oversized copies, never crop.

    Dataset assets, model-visible images and saved renders remain unchanged.
    Pillow's hard decompression-bomb protection remains enabled.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Expected PNG, JPEG or WebP judge evidence")
                size = fit_size(*source.size)
                if size == source.size:
                    return data
                oriented = ImageOps.exif_transpose(source)
                size = fit_size(*oriented.size)
                mode = "RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB"
                resized = oriented.convert(mode).resize(size, Image.Resampling.LANCZOS)
                output = io.BytesIO()
                resized.save(output, format="PNG")
                return output.getvalue()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or unsafe judge image") from exc

"""Bounded in-memory image uploads shared by the browser app and inference API."""

import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

from flask import Request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


class MemoryOnlyRequest(Request):
    def _get_file_stream(
        self, total_content_length, content_type, filename=None, content_length=None
    ):
        return io.BytesIO()


class UploadError(ValueError):
    def __init__(self, message, status=422):
        self.status = status
        super().__init__(message)


@dataclass(frozen=True)
class UploadedImage:
    filename: str
    content: bytes
    sha256: str
    format: str
    width: int
    height: int


def validate_upload(storage, max_bytes=5 * 1024 * 1024, max_pixels=4_000_000):
    if storage is None or not storage.filename:
        raise UploadError("Choose a JPEG or PNG image first.", 400)
    name = secure_filename(storage.filename)
    extension = Path(name).suffix.lower()
    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
    if extension not in formats:
        raise UploadError("Only JPEG (.jpg, .jpeg) and PNG (.png) files are supported.", 415)
    content = storage.stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise UploadError(f"The image must be no larger than {max_bytes // 1024**2} MB.", 413)
    if not content:
        raise UploadError("The image file is empty.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                detected = image.format
                if width * height > max_pixels:
                    raise UploadError(f"Resize the image to {max_pixels:,} pixels or fewer.")
                if detected != formats[extension]:
                    raise UploadError(
                        "The image contents do not match its filename extension.", 415
                    )
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        if isinstance(error, UploadError):
            raise
        raise UploadError("This file could not be decoded as a valid image.") from error
    name = (Path(name).stem[:180] or "image") + extension
    return UploadedImage(
        name, content, hashlib.sha256(content).hexdigest(), detected, width, height
    )

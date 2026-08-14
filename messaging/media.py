"""Download and classify inbound Sendblue media for the local model."""

from __future__ import annotations

import base64
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

MediaKind = Literal["image", "audio", "video"]

MULTIMODAL_REFUSAL = "I don't have multi-modal capabilities."
MAX_MEDIA_BYTES = 15 * 1024 * 1024
_DOWNLOAD_TIMEOUT_S = 30.0
_CONVERT_TIMEOUT_S = 30.0

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif",
               ".webp", ".heic", ".heif", ".bmp", ".tga"}
_AUDIO_EXTS = {".m4a", ".mp3", ".aac", ".caf", ".wav", ".flac"}
_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm"}
_SKIP_EXTS = {".pdf", ".vcf", ".vcard"}


class MediaDownloadError(Exception):
    """Failed to fetch inbound media from Sendblue."""


class UnsupportedAttachment(Exception):
    """Attachment is not image/audio/video (PDF, vCard, unknown)."""


@dataclass(frozen=True)
class DownloadedMedia:
    kind: MediaKind
    mime: str
    data: bytes

    def to_content_part(self) -> dict[str, object]:
        b64 = base64.b64encode(self.data).decode("ascii")
        if self.kind == "image":
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{self.mime};base64,{b64}"},
            }
        if self.kind == "audio":
            return {"type": "input_audio", "input_audio": {"data": b64}}
        return {"type": "input_video", "input_video": {"data": b64}}


def download_media(url: str) -> DownloadedMedia:
    """GET a Sendblue CDN URL, classify it, and convert HEIC/CAF when needed."""
    mime, data = _get_bytes(url)
    kind = classify_media(mime, url, data)
    if kind is None:
        raise UnsupportedAttachment(f"unsupported attachment: {mime or url}")

    if kind == "image" and _is_heic(mime, url, data):
        data, mime = _convert_heic(data)
    elif kind == "audio" and _is_caf(mime, url):
        data, mime = _convert_caf(data)

    return DownloadedMedia(kind=kind, mime=mime, data=data)


def classify_media(mime: str, url: str, data: bytes = b"") -> MediaKind | None:
    mime = (mime or "").split(";")[0].strip().lower()
    if mime in {"application/pdf", "text/vcard", "text/x-vcard"}:
        return None
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"

    ext = Path(urlsplit(url).path).suffix.lower()
    if ext in _SKIP_EXTS:
        return None
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"

    return _sniff_kind(data)


def _get_bytes(url: str) -> tuple[str, bytes]:
    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and int(length) > MAX_MEDIA_BYTES:
                    raise MediaDownloadError(
                        f"media too large ({length} bytes, max {MAX_MEDIA_BYTES})"
                    )
                mime = (response.headers.get("content-type")
                        or "").split(";")[0].strip()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_MEDIA_BYTES:
                        raise MediaDownloadError(
                            f"media too large (max {MAX_MEDIA_BYTES} bytes)"
                        )
                    chunks.append(chunk)
                return mime, b"".join(chunks)
    except MediaDownloadError:
        raise
    except httpx.HTTPError as exc:
        raise MediaDownloadError(f"failed to download media: {exc}") from exc
    except ValueError as exc:
        raise MediaDownloadError(f"invalid content-length: {exc}") from exc


def _sniff_kind(data: bytes) -> MediaKind | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return "image"
    if len(data) < 12:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio"
    if data[:3] == b"ID3" or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "audio"
    if data[4:8] == b"ftyp":
        brand = data[8:12].decode("latin-1", errors="ignore").lower()
        if brand in {"heic", "heix", "heif", "mif1", "msf1"}:
            return "image"
        if brand in {"m4a ", "m4b ", "m4p "}:
            return "audio"
        return "video"
    return None


def _is_heic(mime: str, url: str, data: bytes) -> bool:
    mime = mime.lower()
    if "heic" in mime or "heif" in mime:
        return True
    ext = Path(urlsplit(url).path).suffix.lower()
    if ext in {".heic", ".heif"}:
        return True
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].decode("latin-1", errors="ignore").lower()
        return brand in {"heic", "heix", "heif", "mif1", "msf1"}
    return False


def _is_caf(mime: str, url: str) -> bool:
    return mime.lower() in {"audio/x-caf", "audio/caf"} or Path(
        urlsplit(url).path
    ).suffix.lower() == ".caf"


def _convert_heic(data: bytes) -> tuple[bytes, str]:
    sips = shutil.which("sips")
    if not sips:
        logger.warning("sips not found; sending HEIC as-is")
        return data, "image/heic"
    converted = _run_converter(
        data,
        input_name="in.heic",
        output_name="out.jpg",
        argv_for=lambda src, dst: [sips, "-s",
                                   "format", "jpeg", src, "--out", dst],
    )
    if converted is None:
        return data, "image/heic"
    return converted, "image/jpeg"


def _convert_caf(data: bytes) -> tuple[bytes, str]:
    afconvert = shutil.which("afconvert")
    if not afconvert:
        logger.warning("afconvert not found; sending CAF as-is")
        return data, "audio/x-caf"
    converted = _run_converter(
        data,
        input_name="in.caf",
        output_name="out.wav",
        argv_for=lambda src, dst: [afconvert,
                                   "-f", "WAVE", "-d", "LEI16", src, dst],
    )
    if converted is None:
        return data, "audio/x-caf"
    return converted, "audio/wav"


def _run_converter(
    data: bytes,
    *,
    input_name: str,
    output_name: str,
    argv_for: Callable[[str, str], list[str]],
) -> bytes | None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / input_name
        dst = Path(tmp) / output_name
        src.write_bytes(data)
        cmd = argv_for(str(src), str(dst))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=_CONVERT_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("%s convert failed", cmd[0])
            return None
        if result.returncode != 0 or not dst.exists():
            logger.warning(
                "%s convert failed (rc=%s): %s",
                cmd[0],
                result.returncode,
                (result.stderr or b"").decode("utf-8", errors="replace")[:300],
            )
            return None
        return dst.read_bytes()

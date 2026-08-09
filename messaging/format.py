"""Outbound message formatting for iMessage (plain text, length limits)."""

from __future__ import annotations

import re

# Sendblue / iMessage practical content limit for a single outbound message.
DEFAULT_CHUNK_SIZE = 2000

_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___|\*\*|__|\*|_)(.*?)\1", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.*?)~~", re.DOTALL)
_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*([-*_]\s*){3,}\s*$", re.MULTILINE)
_LIST_MARKER_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+", re.MULTILINE)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def strip_markdown(text: str) -> str:
    """Convert common Markdown to plain text suitable for iMessage."""
    if not text:
        return ""

    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _CODE_FENCE_RE.sub(lambda m: m.group(1).rstrip(), out)
    out = _IMAGE_RE.sub(lambda m: m.group(1) or m.group(2), out)
    out = _LINK_RE.sub(r"\1 (\2)", out)
    out = _INLINE_CODE_RE.sub(r"\1", out)
    out = _HEADING_RE.sub("", out)
    out = _HORIZONTAL_RULE_RE.sub("", out)
    out = _BLOCKQUOTE_RE.sub("", out)
    out = _LIST_MARKER_RE.sub(r"\1", out)
    out = _STRIKE_RE.sub(r"\1", out)
    # Repeat for nested emphasis like ***bold italic***.
    for _ in range(3):
        out = _BOLD_ITALIC_RE.sub(r"\2", out)
    out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out.strip()


def chunk_text(text: str, max_len: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Split ``text`` into chunks of at most ``max_len`` characters.

    Prefers breaking on paragraph, line, then word boundaries when possible.
    """
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if not text:
        return []

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        window = remaining[:max_len]
        split_at: int | None = None
        sep_len = 0
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx > 0:
                split_at = idx
                sep_len = len(sep)
                break
        if split_at is None:
            split_at = max_len
            sep_len = 0

        piece = remaining[:split_at].rstrip()
        if sep_len:
            remaining = remaining[split_at + sep_len :].lstrip()
        else:
            remaining = remaining[split_at:]
        if piece:
            chunks.append(piece)

    return chunks

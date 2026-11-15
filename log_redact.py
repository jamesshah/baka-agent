"""Strip phone numbers and other personal info from log output.

Applied to every logging handler the process emits, and to string fields in
the LLM perf JSONL file. Free-text that is not a phone, email, SSN, or card
number is left in place.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

PHONE = "[phone]"
EMAIL = "[email]"
SSN = "[ssn]"
CARD = "[card]"

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

# Local part, then a dotted domain. Applied before phone patterns so a "+"
# inside an address is not treated as the start of a phone number.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# "+" then digits with optional phone punctuation. Digit count is checked
# in the replacer so a short token like "+1" survives.
_PLUS_PHONE_RE = re.compile(r"\+(?:[\d\s().\-]){9,40}\d")
# NANP with separators or parentheses, optional leading country code 1.
_NANP_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+?1[\s.\-]?)?"
    r"(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}"
    r"(?!\d)"
)
# Bare US numbers. 10-digit values must start with 2-9 so Unix timestamps
# (currently 17xxxxxxxxx) are not treated as phones. 11-digit values are the
# country code plus a 10-digit number.
_BARE_PHONE_RE = re.compile(r"(?<!\d)(?:1[2-9]\d{9}|[2-9]\d{9})(?!\d)")
# 13-19 digits, optional single spaces or dashes. Kept only when Luhn passes.
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")


def redact_text(text: str) -> str:
    """Replace personal info in ``text`` with stable placeholders."""
    if not text:
        return text
    redacted = _EMAIL_RE.sub(EMAIL, text)
    redacted = _SSN_RE.sub(SSN, redacted)
    redacted = _PLUS_PHONE_RE.sub(_plus_phone, redacted)
    redacted = _NANP_RE.sub(PHONE, redacted)
    redacted = _BARE_PHONE_RE.sub(PHONE, redacted)
    return _CARD_RE.sub(_card, redacted)


def redact_data(value: Any) -> Any:
    """Return a copy of ``value`` with personal info removed from strings."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_data(item) for item in value]
    return value


class RedactingFormatter(logging.Formatter):
    """Format with ``inner``, then strip personal info from the full line."""

    def __init__(self, inner: logging.Formatter) -> None:
        super().__init__()
        self.inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(self.inner.format(record))


def configure_logging() -> None:
    """Install the process log format and redact every handler's output.

    Uvicorn sets the root level to WARNING, and importing Composio then
    attaches a handler. ``basicConfig`` does nothing once a handler exists,
    so the level has to be set explicitly or application INFO lines are dropped.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    root.setLevel(logging.INFO)
    install_redacting_formatters()


def install_redacting_formatters(
    loggers: Iterable[logging.Logger] | None = None,
) -> None:
    """Wrap formatters so personal info never reaches a log file."""
    targets = list(loggers) if loggers is not None else _all_loggers()
    for logger in targets:
        for handler in logger.handlers:
            current = handler.formatter
            if isinstance(current, RedactingFormatter):
                continue
            if current is None:
                current = logging.Formatter(_LOG_FORMAT)
            handler.setFormatter(RedactingFormatter(current))


def _all_loggers() -> list[logging.Logger]:
    manager = logging.Logger.manager
    found = [logging.getLogger()]
    for candidate in list(manager.loggerDict.values()):
        if isinstance(candidate, logging.Logger):
            found.append(candidate)
    return found


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def _plus_phone(match: re.Match[str]) -> str:
    digits = _digits(match.group(0))
    if 10 <= len(digits) <= 15:
        return PHONE
    return match.group(0)


def _card(match: re.Match[str]) -> str:
    digits = _digits(match.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return CARD
    return match.group(0)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = ord(char) - 48
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0

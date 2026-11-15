"""Personal info is removed from log lines and the perf JSONL file."""

from __future__ import annotations

import io
import json
import logging
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm.perf import log_completion
from log_redact import (
    CARD,
    EMAIL,
    PHONE,
    SSN,
    RedactingFormatter,
    configure_logging,
    install_redacting_formatters,
    redact_data,
    redact_text,
)


class RedactTextTests(unittest.TestCase):
    def test_e164_and_formatted_us_numbers(self) -> None:
        text = (
            "inbound from +15551234567 and +1 (555) 987-6543 "
            "or 555-000-1111 or 555.000.2222"
        )
        redacted = redact_text(text)
        self.assertNotIn("555", redacted)
        self.assertEqual(redacted.count(PHONE), 4)

    def test_bare_us_numbers(self) -> None:
        redacted = redact_text("allow 5551234567 and 15551234567")
        self.assertEqual(redacted, f"allow {PHONE} and {PHONE}")

    def test_unix_timestamp_and_short_counts_stay(self) -> None:
        text = "ts=1731283200 indexed 12 skill(s) port 8000 +1"
        self.assertEqual(redact_text(text), text)

    def test_email_ssn_and_card(self) -> None:
        text = (
            "mail ada@example.com ssn 123-45-6789 "
            "card 4111111111111111 amex 378282246310005"
        )
        redacted = redact_text(text)
        self.assertEqual(
            redacted,
            f"mail {EMAIL} ssn {SSN} card {CARD} amex {CARD}",
        )

    def test_non_card_long_number_stays(self) -> None:
        text = "id 1234567890123456"
        self.assertEqual(redact_text(text), text)

    def test_email_with_plus_is_one_email(self) -> None:
        self.assertEqual(
            redact_text("reach ada+tag@example.com"),
            f"reach {EMAIL}",
        )

    def test_nested_strings_only(self) -> None:
        payload = {
            "session_id": "+15551234567",
            "prompt_tokens": 4852,
            "label": "chat",
        }
        redacted = redact_data(payload)
        self.assertEqual(redacted["session_id"], PHONE)
        self.assertEqual(redacted["prompt_tokens"], 4852)
        self.assertEqual(payload["session_id"], "+15551234567")


class LoggingRedactTests(unittest.TestCase):
    def test_formatter_redacts_message_and_traceback(self) -> None:
        formatter = RedactingFormatter(logging.Formatter("%(message)s"))
        try:
            raise RuntimeError("callback for +15559876543 failed")
        except RuntimeError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed to send to %s",
                args=("+15551234567",),
                exc_info=sys.exc_info(),
            )
        line = formatter.format(record)
        self.assertNotIn("+1555", line)
        self.assertIn(PHONE, line)
        self.assertIn("failed to send to", line)

    def test_install_wraps_only_the_given_logger(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("baka.tests.log_redact")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        try:
            install_redacting_formatters([logger])
            install_redacting_formatters([logger])
            logger.info("inbound from %s: email %s", "+15551234567", "a@b.co")
            output = stream.getvalue()
            self.assertIn(PHONE, output)
            self.assertIn(EMAIL, output)
            self.assertNotIn("+15551234567", output)
            self.assertNotIn("a@b.co", output)
            self.assertEqual(
                sum(isinstance(item, RedactingFormatter) for item in (handler.formatter,)),
                1,
            )
        finally:
            logger.removeHandler(handler)

    def test_configure_logging_keeps_info_when_a_handler_already_exists(self) -> None:
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        stream = io.StringIO()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        root.addHandler(logging.StreamHandler(stream))
        try:
            configure_logging()
            logging.getLogger("baka.tests.preexisting").info("call %s", "+15551234567")
            output = stream.getvalue()
            self.assertIn(PHONE, output)
            self.assertNotIn("+15551234567", output)
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in saved_handlers:
                root.addHandler(handler)
            root.setLevel(saved_level)

    def test_configure_logging_is_idempotent_without_handlers(self) -> None:
        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        root.handlers.clear()
        try:
            configure_logging()
            configure_logging()
            wrapped = [
                handler.formatter
                for handler in root.handlers
                if isinstance(handler.formatter, RedactingFormatter)
            ]
            self.assertEqual(len(wrapped), len(root.handlers))
            self.assertGreaterEqual(len(wrapped), 1)
        finally:
            for handler in list(root.handlers):
                if handler not in saved_handlers:
                    root.removeHandler(handler)
            root.setLevel(saved_level)


class PerfLogRedactTests(unittest.TestCase):
    def test_jsonl_session_phone_is_redacted(self) -> None:
        data = {
            "choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm-perf.jsonl"
            metrics = log_completion(
                data,
                wall_ms=10.0,
                path=path,
                session_id="+15551234567",
                label="chat",
            )
            parsed = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["session_id"], PHONE)
            self.assertEqual(parsed["prompt_tokens"], 12)
            self.assertEqual(parsed["label"], "chat")
            self.assertEqual(metrics["session_id"], "+15551234567")


if __name__ == "__main__":
    unittest.main()

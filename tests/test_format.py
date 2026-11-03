from __future__ import annotations

import unittest

from messaging.format import strip_markdown

WHOOP_URL = (
    "https://api.prod.whoop.com/oauth/oauth2/auth?response_type=code"
    "&client_id=75850f35-d8d9-4f8d-a523-fe939acd288b"
    "&redirect_uri=https%3A%2F%2Fbaka-agent.jamesshah.tech%2Foauth%2Fwhoop%2Fcallback"
    "&scope=offline+read%3Arecovery&state=RRZS-qyddHZhm65vYwPO2A"
)


class StripMarkdownTests(unittest.TestCase):
    def test_bare_url_keeps_underscores(self) -> None:
        text = f"Here's the link:\n\n{WHOOP_URL}\n\nOpen it in your browser."
        self.assertIn(WHOOP_URL, strip_markdown(text))

    def test_markdown_link_url_keeps_underscores(self) -> None:
        out = strip_markdown(f"**Tap** [link Whoop]({WHOOP_URL})")
        self.assertEqual(out, f"Tap link Whoop ({WHOOP_URL})")

    def test_snake_case_words_survive(self) -> None:
        self.assertEqual(
            strip_markdown("set recovery_score and sleep_hours"),
            "set recovery_score and sleep_hours",
        )

    def test_emphasis_still_stripped(self) -> None:
        self.assertEqual(
            strip_markdown("**bold**, *italic*, _under_, __strong__, ***both***"),
            "bold, italic, under, strong, both",
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import tempfile
from typing import List, Dict, Any

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import caption_engine


class TestCaptionAesthetics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_header_typography_and_mobile_safe_zone(self):
        """Verifies font, sizing, colors, heavy outline, deep shadow, and MarginV=580 in ASS header."""
        header = caption_engine._ass_header()

        # 1. High impact bold sans-serif font
        self.assertIn("Style: Default,Arial Black,84", header)

        # 2. Colors: Primary=Yellow (&H0000FFFF), Secondary=White (&H00FFFFFF), Outline=Black (&H00000000), Shadow=Deep (&H80000000)
        self.assertIn("&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000", header)

        # 3. Heavy 4.5px outline stroke and 3px deep shadow with Alignment=2 (bottom centered)
        self.assertIn("1,4.5,3,2,", header)

        # 4. Mobile safe zone: MarginV=580
        self.assertIn(",580,1", header)

    def test_15_word_kinetic_sample_generation(self):
        """Generates an ASS script from a 15-word transcript and verifies centiseconds & phrase length."""
        out_path = os.path.join(self.temp_dir.name, "test_15words.ass")

        # 15 words spoken over 6 seconds
        raw_words = [
            "This", "is", "a", "high", "impact",
            "kinetic", "typography", "engine", "built", "for",
            "short", "form", "viral", "video", "content."
        ]
        sample_words = []
        t = 0.0
        for w in raw_words:
            sample_words.append({
                "word": w,
                "start": round(t, 2),
                "end": round(t + 0.35, 2)
            })
            t += 0.40

        res_path = caption_engine.generate_karaoke_ass(
            words=sample_words,
            output_ass_path=out_path,
        )

        self.assertIsNotNone(res_path)
        self.assertTrue(os.path.isfile(out_path))

        with open(out_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        dialogue_lines = [l.strip() for l in lines if l.startswith("Dialogue:")]
        self.assertGreater(len(dialogue_lines), 0)

        # Verify on-screen burst length: each line should contain 2 to 4 words
        for d in dialogue_lines:
            # Count word occurrences or \kf tags in dialogue line
            kf_count = d.count(r"\kf")
            self.assertGreaterEqual(kf_count, 1)
            self.assertLessEqual(kf_count, 4)

            # Check centisecond timing tags
            # Each word duration was 0.35s -> ~35 centiseconds
            self.assertIn(r"\kf", d)

            # Check that Dialogue timing conforms to H:MM:SS.cs
            parts = d.split(",")
            start_ts = parts[1]
            end_ts = parts[2]
            self.assertRegex(start_ts, r"^\d+:\d{2}:\d{2}\.\d{2}$")
            self.assertRegex(end_ts, r"^\d+:\d{2}:\d{2}\.\d{2}$")

    def test_word_timing_centiseconds_accuracy(self):
        """Verifies that centiseconds accurately correspond to spoken word duration."""
        out_path = os.path.join(self.temp_dir.name, "test_timing.ass")
        words = [
            {"word": "Quick", "start": 1.00, "end": 1.50},    # 0.50s = 50cs
            {"word": "brown", "start": 1.50, "end": 2.25},    # 0.75s = 75cs
            {"word": "fox", "start": 2.25, "end": 2.50},      # 0.25s = 25cs
        ]

        caption_engine.generate_karaoke_ass(words, out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(r"\kf50", content)
        self.assertIn(r"\kf75", content)
        self.assertIn(r"\kf25", content)

    def test_clean_fallback_on_empty_words(self):
        """Ensures empty words array returns None and does not create an empty file."""
        out_path = os.path.join(self.temp_dir.name, "empty_should_not_exist.ass")

        # Test empty list
        res = caption_engine.generate_karaoke_ass([], out_path)
        self.assertIsNone(res)
        self.assertFalse(os.path.exists(out_path))

        # Test None
        res_none = caption_engine.generate_karaoke_ass(None, out_path)
        self.assertIsNone(res_none)
        self.assertFalse(os.path.exists(out_path))

        # Test whitespace-only words
        res_whitespace = caption_engine.generate_karaoke_ass([{"word": "   "}], out_path)
        self.assertIsNone(res_whitespace)
        self.assertFalse(os.path.exists(out_path))


    def test_profanity_censoring_and_scunthorpe_protection(self):
        """Verifies profanity is masked with asterisks while common English substrings are unharmed."""
        # Profane words should be censored
        self.assertEqual(caption_engine.censor_profanity("fuck"), "f***")
        self.assertEqual(caption_engine.censor_profanity("fucking"), "f***")
        self.assertEqual(caption_engine.censor_profanity("shit"), "sh*t")
        self.assertEqual(caption_engine.censor_profanity("bitch"), "b***h")
        self.assertEqual(caption_engine.censor_profanity("ass"), "a**")
        self.assertEqual(caption_engine.censor_profanity("asshole"), "a**hole")

        # Harmless words containing profanity substrings must NOT be censored
        self.assertEqual(caption_engine.censor_profanity("classic"), "classic")
        self.assertEqual(caption_engine.censor_profanity("assistant"), "assistant")
        self.assertEqual(caption_engine.censor_profanity("assessment"), "assessment")
        self.assertEqual(caption_engine.censor_profanity("passion"), "passion")
        self.assertEqual(caption_engine.censor_profanity("class"), "class")

        # In karaoke generation, censored text appears in ASS
        out_path = os.path.join(self.temp_dir.name, "test_censor.ass")
        words = [
            {"word": "This", "start": 0.0, "end": 0.3},
            {"word": "fucking", "start": 0.3, "end": 0.6},
            {"word": "class", "start": 0.6, "end": 1.0},
        ]
        caption_engine.generate_karaoke_ass(words, out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("f***", content)
        self.assertIn("class", content)
        self.assertNotIn("fucking", content)


if __name__ == "__main__":
    unittest.main()

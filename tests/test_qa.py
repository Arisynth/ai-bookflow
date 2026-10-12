from __future__ import annotations

import unittest

from bookflow.epub import Segment
from bookflow.qa import deterministic_qa


class QualityAssuranceTests(unittest.TestCase):
    def test_numeric_formatting_accepts_leading_zero_and_date_translation(self):
        segment = Segment("id", "chapter-1", "Compiled on 08/26/2026", "chapter.xhtml", 1)
        report = deterministic_qa([segment], {"id": "编译于 2026 年 8 月 26 日"})
        self.assertTrue(report.ok, report.issues)

    def test_numeric_value_change_is_reported(self):
        segment = Segment("id", "chapter-1", "There are 8 chapters", "chapter.xhtml", 1)
        report = deterministic_qa([segment], {"id": "共有 9 章"})
        self.assertFalse(report.ok)
        self.assertIn("8", report.issues[0])

    def test_explicit_scale_conversion_is_not_an_omission(self):
        segment = Segment("id", "c", "1.3 billion phones, 2 million users, 12,345,678 items", "c.xhtml", 0)
        result = deterministic_qa([segment], {"id": "13亿部手机、200 万用户、12,345,678件物品"})
        self.assertTrue(result.ok, result.issues)

    def test_scale_conversion_does_not_hide_wrong_order_of_magnitude(self):
        segment = Segment("id", "c", "1.3 billion phones", "c.xhtml", 0)
        for translated in ("13部手机", "1.3亿部手机", "13 万部手机"):
            with self.subTest(translated=translated):
                self.assertFalse(deterministic_qa([segment], {"id": translated}).ok)

    def test_chinese_number_words_and_ordinals_are_recognized(self):
        segment = Segment(
            "id", "c", "In year 1, System 2 was used by 100% of 1 in 100,000,000.", "c.xhtml", 0
        )
        translated = "第一年，系统二的使用率为百分之百，概率为一亿分之一。"
        self.assertTrue(deterministic_qa([segment], {"id": translated}).ok)

    def test_bibliographic_url_digits_are_ignored(self):
        segment = Segment(
            "id", "c", "See 2013: https://example.com/SB10001424127887323539804578261780648285770.", "c.xhtml", 0
        )
        translated = "参见2013年：https://example.com/SB10001424127887323539804578261780648285770。"
        self.assertTrue(deterministic_qa([segment], {"id": translated}).ok)

    def test_chinese_decades_and_colloquial_odds_are_recognized(self):
        segment = Segment("id", "c", "In the 1980s, the odds were 40 to 60.", "c.xhtml", 0)
        translated = "在20世纪80年代，概率大约是四成比六成。"
        self.assertTrue(deterministic_qa([segment], {"id": translated}).ok)

    def test_chinese_trillion_scale_is_recognized(self):
        segment = Segment("id", "c", "a $2 trillion risk", "c.xhtml", 0)
        translated = "价值2万亿美元的风险"
        self.assertTrue(deterministic_qa([segment], {"id": translated}).ok)


if __name__ == "__main__":
    unittest.main()

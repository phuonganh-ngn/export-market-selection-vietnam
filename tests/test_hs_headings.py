import unittest

from common import hs2_chapter_descriptions


class HSChapterHeadingTests(unittest.TestCase):
    def test_official_hs12_headings_cover_active_chapters(self):
        headings = hs2_chapter_descriptions()
        self.assertEqual(len(headings), 96)
        self.assertEqual(len(set(headings)), 96)
        self.assertNotIn("77", headings)
        self.assertEqual(headings["09"], "Coffee, tea, maté and spices.")


if __name__ == "__main__":
    unittest.main()

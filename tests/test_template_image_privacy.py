import unittest
from pathlib import Path

from scripts.sanitize_template_images import find_unsanitized_images


class TemplateImagePrivacyTest(unittest.TestCase):
    def test_only_annotated_template_regions_are_visible(self):
        coco_path = Path(__file__).parents[1] / "assets" / "coco_annotations.json"
        self.assertEqual([], find_unsanitized_images(coco_path))


if __name__ == "__main__":
    unittest.main()

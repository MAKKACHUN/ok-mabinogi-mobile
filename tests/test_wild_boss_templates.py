import json
import unittest
from pathlib import Path


class WildBossTemplateTest(unittest.TestCase):
    def test_numbered_template_annotations_are_complete(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data = json.loads(
            (project_root / "assets" / "coco_annotations.json").read_text(
                encoding="utf-8-sig"
            )
        )
        categories = {
            category["id"]: category["name"]
            for category in data["categories"]
        }
        images = {
            image["id"]: image["file_name"]
            for image in data["images"]
        }
        annotations = {
            categories[annotation["category_id"]]: annotation
            for annotation in data["annotations"]
            if categories[annotation["category_id"]]
            in self.expected_bboxes()
        }

        self.assertEqual(
            set(annotations),
            set(self.expected_bboxes()),
        )
        for feature_name, expected_bbox in self.expected_bboxes().items():
            annotation = annotations[feature_name]
            self.assertEqual(annotation["bbox"], expected_bbox)
            image_path = project_root / "assets" / images[
                annotation["image_id"]
            ]
            self.assertTrue(image_path.is_file(), feature_name)

    def test_old_wild_boss_categories_are_removed(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        data = json.loads(
            (project_root / "assets" / "coco_annotations.json").read_text(
                encoding="utf-8-sig"
            )
        )
        category_names = {
            category["name"] for category in data["categories"]
        }
        self.assertFalse(
            category_names
            & {
                "wild_boss_menu",
                "wild_boss_move_confirm",
                "wild_boss_entry_confirm",
                "wild_boss_entry_not_ready",
                "wild_boss_waiting",
                "wild_boss_room_ready",
                "wild_boss_victory",
                "wild_boss_skip_cutscene",
                "wild_boss_exit_task",
                "wild_boss_exit_confirm",
            }
        )

    @staticmethod
    def expected_bboxes() -> dict[str, list[int]]:
        return {
            "16_01": [706, 529, 181, 24],
            "16_02": [885, 765, 54, 26],
            "17_01": [761, 185, 79, 20],
            "18_01": [1498, 253, 81, 28],
            "19_01": [699, 642, 195, 26],
            "20_01": [1440, 288, 106, 19],
            "28_01": [1144, 481, 70, 63],
            "29_01": [562, 536, 478, 24],
            "29_02": [773, 767, 54, 26],
            "30_01": [723, 572, 149, 30],
            "30_02": [915, 767, 54, 26],
            "31_01": [1543, 286, 35, 18],
            "31_02": [1502, 311, 37, 20],
            "33_01": [1430, 288, 107, 18],
            "34_01": [1429, 288, 107, 19],
            "35_01": [1438, 35, 102, 25],
            "36_01": [761, 185, 78, 20],
        }


if __name__ == "__main__":
    unittest.main()

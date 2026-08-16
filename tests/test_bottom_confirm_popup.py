import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2

from src.plugins.wild_boss.tasks.AutoWildBossTask import AutoWildBossTask
from src.tasks.BaseDNATask import BaseDNATask


class BottomConfirmPopupTest(unittest.TestCase):
    def make_task(self) -> BaseDNATask:
        task = BaseDNATask.__new__(BaseDNATask)
        task._bottom_confirm_check_enabled = True
        task._bottom_confirm_last_handled_at = float("-inf")
        task.find_one = MagicMock(return_value=object())
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()
        task.find_bottom_confirm_color_button = MagicMock(return_value=False)
        return task

    def test_detected_popup_presses_space(self) -> None:
        task = self.make_task()

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            task.sleep_check()

        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            [
                "bottom_confirm_item_delivery_text",
                "bottom_confirm_button",
            ],
        )
        for find_call in task.find_one.call_args_list:
            self.assertNotIn("use_gray_scale", find_call.kwargs)
        task.find_one.assert_any_call(
            "bottom_confirm_button",
            horizontal_variance=0.03,
            vertical_variance=0.03,
            threshold=0.80,
            canny_lower=50,
            canny_higher=150,
        )
        task.ensure_in_front.assert_called_once_with()
        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=0,
        )
        self.assertTrue(task._bottom_confirm_did_press)

    def test_visible_popup_is_not_repeated_inside_retry_window(self) -> None:
        task = self.make_task()

        with patch(
            "src.tasks.BaseDNATask.time.monotonic",
            side_effect=[10.0, 11.0, 12.1],
        ):
            task.sleep_check()
            task.sleep_check()
            task.sleep_check()

        self.assertEqual(task.send_key.call_count, 2)

    def test_missing_popup_does_not_press_space(self) -> None:
        task = self.make_task()
        task.find_one.return_value = None

        task.sleep_check()

        task.send_key.assert_not_called()

    def test_confirm_button_without_delivery_text_does_not_press(self) -> None:
        task = self.make_task()
        task.find_one.return_value = None
        task.find_bottom_confirm_color_button.return_value = True

        self.assertFalse(task.handle_bottom_confirm_popup())

        task.find_bottom_confirm_color_button.assert_not_called()
        task.send_key.assert_not_called()

    def test_green_button_fallback_presses_space(self) -> None:
        task = self.make_task()
        task.find_one.side_effect = [object(), None]
        task.find_bottom_confirm_color_button.return_value = True

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            self.assertTrue(task.handle_bottom_confirm_popup())

        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=0,
        )

    def test_green_button_fallback_matches_37_but_not_entry_dialog(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        popup = cv2.imread(
            str(project_root / "assets" / "images" / "bottom_confirm_button.png")
        )
        entry_dialog = cv2.imread(
            str(project_root / "assets" / "images" / "0.png")
        )
        task = BaseDNATask.__new__(BaseDNATask)

        self.assertTrue(task.find_bottom_confirm_color_button(popup))
        self.assertFalse(task.find_bottom_confirm_color_button(entry_dialog))

    def test_bottom_confirm_preserves_battle_victory_state(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task._bottom_confirm_check_enabled = True
        task._bottom_confirm_last_handled_at = float("-inf")
        task._monitoring_battle_completion = True
        task.find_one = MagicMock(return_value=object())
        task.find_bottom_confirm_color_button = MagicMock(return_value=False)
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            self.assertTrue(task.handle_bottom_confirm_popup())

        self.assertTrue(task._battle_victory_confirmed)
        self.assertEqual(task._battle_confirm_handled_at, 10.0)
        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=0,
        )

    def test_any_bottom_confirm_during_battle_records_recent_time(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task._bottom_confirm_check_enabled = True
        task._bottom_confirm_last_handled_at = float("-inf")
        task._monitoring_battle_completion = True
        task._battle_victory_confirmed = False
        task.find_one = MagicMock(return_value=object())
        task.find_bottom_confirm_color_button = MagicMock(return_value=False)
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            self.assertTrue(task.handle_bottom_confirm_popup())

        self.assertTrue(task._battle_victory_confirmed)
        self.assertEqual(task._battle_confirm_handled_at, 10.0)

    def test_boss_readiness_handles_popup_before_state_checks(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task.handle_bottom_confirm_popup = MagicMock(return_value=True)
        task.is_main_screen = MagicMock()
        task.find_one = MagicMock()

        self.assertFalse(task.is_boss_ready_to_fight())
        task.is_main_screen.assert_not_called()
        task.find_one.assert_not_called()

    def test_battle_monitor_handles_popup_before_victory_checks(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task.handle_bottom_confirm_popup = MagicMock(return_value=True)
        task.find_one = MagicMock(return_value=None)

        self.assertIsNone(task.find_battle_completion_or_handle_overlay())
        task.find_one.assert_called_once_with(
            task.EXIT_CONFIRM_FEATURE,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )

    def test_bottom_confirm_allows_exit_task_click(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task._bottom_confirm_check_enabled = False
        task._battle_victory_confirmed = True
        exit_task = object()
        task.find_one = MagicMock(
            side_effect=[None, None, exit_task]
        )
        task.find_any_feature = MagicMock(return_value=None)
        task.move_and_click = MagicMock()
        task.pydirect_interaction = MagicMock()
        task.sleep = MagicMock()
        task.log_info = MagicMock()
        task._executor = MagicMock()
        task._executor.method.width = 1600
        task._executor.method.height = 900

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(result)
        task.move_and_click.assert_called_once_with(
            exit_task,
            after_sleep=0.1,
        )

    def test_unconfirmed_victory_wait_log_is_rate_limited(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task._bottom_confirm_check_enabled = False
        task._battle_victory_confirmed = False
        task._last_exit_wait_log_at = float("-inf")
        exit_task = object()
        task.find_one = MagicMock(
            side_effect=[
                None,
                None,
                exit_task,
                None,
                None,
                exit_task,
            ]
        )
        task.find_any_feature = MagicMock(return_value=None)
        task.log_info = MagicMock()

        with patch(
            "src.plugins.wild_boss.tasks.AutoWildBossTask.time.monotonic",
            side_effect=[16.0, 17.0],
        ):
            self.assertIsNone(
                task.find_battle_completion_or_handle_overlay()
            )
            self.assertIsNone(
                task.find_battle_completion_or_handle_overlay()
            )

        task.log_info.assert_called_once_with(
            "右側任務欄『的領域』已出現，但仍未處理"
            "36_01／17_01 或底部確認；繼續等候"
        )

    def test_template_annotation_and_image_exist(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        annotation_path = project_root / "assets" / "coco_annotations.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8-sig"))

        button_category = next(
            item
            for item in data["categories"]
            if item["name"] == "bottom_confirm_button"
        )
        delivery_category = next(
            item
            for item in data["categories"]
            if item["name"] == "bottom_confirm_item_delivery_text"
        )
        button_annotation = next(
            item
            for item in data["annotations"]
            if item["category_id"] == button_category["id"]
        )
        delivery_annotation = next(
            item
            for item in data["annotations"]
            if item["category_id"] == delivery_category["id"]
        )
        image = next(
            item
            for item in data["images"]
            if item["id"] == button_annotation["image_id"]
        )

        self.assertEqual(button_annotation["bbox"], [765, 805, 70, 40])
        self.assertEqual(
            delivery_annotation["bbox"],
            [600, 344, 395, 28],
        )
        self.assertEqual(
            delivery_annotation["image_id"],
            button_annotation["image_id"],
        )
        self.assertTrue((project_root / "assets" / image["file_name"]).is_file())


if __name__ == "__main__":
    unittest.main()

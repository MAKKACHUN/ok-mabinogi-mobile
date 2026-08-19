import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ok import BaseTask

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
        task.sleep = MagicMock()
        return task

    def test_detected_popup_presses_space(self) -> None:
        task = self.make_task()

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            task.sleep_check()

        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            ["37_01"],
        )
        for find_call in task.find_one.call_args_list:
            self.assertNotIn("use_gray_scale", find_call.kwargs)
        task.find_one.assert_called_once_with(
            "37_01",
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
        task.sleep.assert_called_once_with(0.8)
        self.assertTrue(task._bottom_confirm_did_press)
        self.assertFalse(task._bottom_confirm_action_guard)

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

    def test_bottom_confirm_preserves_battle_victory_state(self) -> None:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task._bottom_confirm_check_enabled = True
        task._bottom_confirm_last_handled_at = float("-inf")
        task._monitoring_battle_completion = True
        task.find_one = MagicMock(return_value=object())
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()
        task.sleep = MagicMock()

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
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()
        task.sleep = MagicMock()

        with patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0):
            self.assertTrue(task.handle_bottom_confirm_popup())

        self.assertTrue(task._battle_victory_confirmed)
        self.assertEqual(task._battle_confirm_handled_at, 10.0)

    def test_wait_checks_are_limited_to_twice_per_second(self) -> None:
        task = self.make_task()
        task._bottom_confirm_last_scan_at = float("-inf")
        task.handle_bottom_confirm_popup = MagicMock(return_value=False)

        with patch(
            "src.tasks.BaseDNATask.time.monotonic",
            side_effect=[10.0, 10.2, 10.5],
        ):
            task.check_bottom_confirm_during_wait()
            task.check_bottom_confirm_during_wait()
            task.check_bottom_confirm_during_wait()

        self.assertEqual(task.handle_bottom_confirm_popup.call_count, 2)

    def test_wait_resumes_original_condition_after_popup(self) -> None:
        task = self.make_task()
        task.check_bottom_confirm_during_wait = MagicMock(
            side_effect=[True, False]
        )
        original_condition = MagicMock(return_value="ready")

        def run_two_iterations(_task, wrapped_condition, **_kwargs):
            return [wrapped_condition(), wrapped_condition()]

        with patch.object(BaseTask, "wait_until", new=run_two_iterations):
            result = BaseDNATask.wait_until(task, original_condition)

        self.assertEqual(result, [None, "ready"])
        original_condition.assert_called_once_with()

    def test_popup_space_is_protected_by_reentrancy_guard(self) -> None:
        task = self.make_task()
        task.send_key = BaseDNATask.send_key.__get__(task, BaseDNATask)
        task.check_bottom_confirm_before_action = MagicMock(
            wraps=task.check_bottom_confirm_before_action
        )

        with (
            patch.object(BaseTask, "send_key", return_value=None) as parent_send,
            patch("src.tasks.BaseDNATask.time.monotonic", return_value=10.0),
        ):
            self.assertTrue(task.handle_bottom_confirm_popup())

        task.check_bottom_confirm_before_action.assert_called_once_with()
        task.find_one.assert_called_once_with(
            "37_01",
            horizontal_variance=0.03,
            vertical_variance=0.03,
            threshold=0.80,
            canny_lower=50,
            canny_higher=150,
        )
        parent_send.assert_called_once_with(
            "space",
            down_time=0.08,
            interval=-1,
            after_sleep=0,
        )
        self.assertFalse(task._bottom_confirm_action_guard)

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

        delivery_category = next(
            item
            for item in data["categories"]
            if item["name"] == "37_01"
        )
        delivery_annotation = next(
            item
            for item in data["annotations"]
            if item["category_id"] == delivery_category["id"]
        )
        image = next(
            item
            for item in data["images"]
            if item["id"] == delivery_annotation["image_id"]
        )

        self.assertEqual(delivery_annotation["bbox"], [604, 348, 386, 19])
        self.assertTrue((project_root / "assets" / image["file_name"]).is_file())


if __name__ == "__main__":
    unittest.main()

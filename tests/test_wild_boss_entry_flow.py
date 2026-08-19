import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from src.plugins.wild_boss.models import HONG_KONG_TIMEZONE
from src.plugins.wild_boss.tasks.AutoWildBossTask import AutoWildBossTask


class WildBossEntryFlowTest(unittest.TestCase):
    def make_task(self) -> AutoWildBossTask:
        task = AutoWildBossTask.__new__(AutoWildBossTask)
        task.log_info = MagicMock()
        task.ensure_in_front = MagicMock()
        task.send_key = MagicMock()
        task.sleep = MagicMock()
        task.navigate_to_boss = MagicMock(return_value=True)
        task.wait_until_scheduled_time = MagicMock()
        task.wait_until_boss_ready = MagicMock(return_value=True)
        task.next_frame = MagicMock()
        task.pydirect_interaction = MagicMock()
        task.check_bottom_confirm_before_action = MagicMock()
        task.capture_boss_room_exit_fingerprint = MagicMock(
            return_value=True
        )
        task.leave_boss_room = MagicMock(return_value=True)
        task.open_exit_confirmation_for_retry = MagicMock(
            return_value=True
        )
        task._executor = MagicMock()
        task._executor.method.width = 1600
        task._executor.method.height = 900
        return task

    def test_confirm_dialog_enters_room_with_space(self) -> None:
        task = self.make_task()
        task.wait_for_entry_dialog = MagicMock(return_value="confirm")
        task.run_boss_cycle = MagicMock(return_value=True)
        scheduled = datetime(
            2026, 8, 9, 20, 0, tzinfo=HONG_KONG_TIMEZONE
        )
        deadline = scheduled + timedelta(minutes=29)
        task.current_hong_kong_time = MagicMock(
            side_effect=[scheduled, deadline, deadline]
        )

        result = task.execute_boss(
            0,
            "佩里",
            scheduled,
        )

        self.assertTrue(result)
        task.navigate_to_boss.assert_called_once_with(0, "佩里")
        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=3.0,
        )
        task.run_boss_cycle.assert_called_once_with("佩里", scheduled)

    def test_not_ready_waits_then_retries_and_enters(self) -> None:
        task = self.make_task()
        task.wait_for_entry_dialog = MagicMock(
            side_effect=["not_ready", "confirm"]
        )
        task.run_boss_cycle = MagicMock(return_value=True)
        scheduled = datetime(
            2026,
            8,
            9,
            20,
            0,
            tzinfo=HONG_KONG_TIMEZONE,
        )
        deadline = scheduled + timedelta(minutes=29)
        task.current_hong_kong_time = MagicMock(
            side_effect=[
                scheduled - timedelta(minutes=2),
                scheduled,
                deadline,
                deadline,
            ]
        )

        result = task.execute_boss(2, "克拉瑪", scheduled)

        self.assertTrue(result)
        self.assertEqual(task.navigate_to_boss.call_count, 2)
        task.wait_until_scheduled_time.assert_called_once_with(
            scheduled,
            "克拉瑪",
        )
        self.assertEqual(task.send_key.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in task.send_key.call_args_list],
            ["space", "space"],
        )

    def test_stuck_entrance_moves_back_then_retries(self) -> None:
        task = self.make_task()
        task.wait_for_entry_dialog = MagicMock(
            side_effect=["stuck", "confirm"]
        )
        task.run_boss_cycle = MagicMock(return_value=True)
        scheduled = datetime(
            2026, 8, 11, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )
        deadline = scheduled + timedelta(minutes=29)
        task.current_hong_kong_time = MagicMock(
            side_effect=[scheduled, scheduled, deadline, deadline]
        )
        task.recover_from_stuck_entrance = MagicMock()

        result = task.execute_boss(2, "克拉瑪", scheduled)

        self.assertTrue(result)
        self.assertEqual(task.navigate_to_boss.call_count, 2)
        task.recover_from_stuck_entrance.assert_called_once_with("克拉瑪")

    def test_completed_room_retries_exit_instead_of_stuck_recovery(self) -> None:
        task = self.make_task()
        task.wait_for_entry_dialog = MagicMock(
            side_effect=["inside_completed_room", "confirm"]
        )
        task.run_boss_cycle = MagicMock(return_value=True)
        task.recover_from_stuck_entrance = MagicMock()
        scheduled = datetime(
            2026, 8, 11, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )
        deadline = scheduled + timedelta(minutes=29)
        task.current_hong_kong_time = MagicMock(
            side_effect=[scheduled, scheduled, deadline, deadline]
        )

        result = task.execute_boss(2, "克拉瑪", scheduled)

        self.assertTrue(result)
        self.assertEqual(task.navigate_to_boss.call_count, 2)
        task.open_exit_confirmation_for_retry.assert_called_once_with()
        task.leave_boss_room.assert_any_call("克拉瑪")
        task.recover_from_stuck_entrance.assert_not_called()

    def test_static_minimap_for_10_seconds_returns_stuck(self) -> None:
        task = self.make_task()
        task.find_entry_dialog_state = MagicMock(return_value=None)
        task.is_main_screen = MagicMock(return_value=True)
        fingerprint = np.zeros(1920, dtype=np.uint8)
        task.get_minimap_inner_fingerprint = MagicMock(
            return_value=fingerprint
        )

        result = AutoWildBossTask.wait_for_entry_dialog(task)

        self.assertEqual(result, "stuck")
        self.assertEqual(
            task.get_minimap_inner_fingerprint.call_count,
            11,
        )

    def test_completed_room_marker_is_detected_before_stuck_timer(self) -> None:
        task = self.make_task()
        task.handle_bottom_confirm_popup = MagicMock(return_value=False)
        task.find_all_features = MagicMock(side_effect=[None, None])
        task.find_one = MagicMock(return_value=object())

        result = task.find_entry_dialog_state()

        self.assertEqual(result, "inside_completed_room")
        task.find_one.assert_called_once_with(
            task.EXIT_TASK_FEATURE,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )

    def test_minimap_fingerprint_uses_full_inner_circle_only(self) -> None:
        task = self.make_task()
        base = np.zeros((900, 1600, 3), dtype=np.uint8)
        inner_changed = base.copy()
        inner_changed[83:123, 1472:1512] = 255
        outer_changed = base.copy()
        outer_changed[83:123, 1562:1592] = 255

        base_fingerprint = task.get_minimap_inner_fingerprint(base)
        inner_fingerprint = task.get_minimap_inner_fingerprint(
            inner_changed
        )
        outer_fingerprint = task.get_minimap_inner_fingerprint(
            outer_changed
        )

        self.assertGreater(base_fingerprint.size, 10000)
        self.assertGreater(
            float(np.mean(np.abs(inner_fingerprint - base_fingerprint))),
            0,
        )
        np.testing.assert_array_equal(
            outer_fingerprint,
            base_fingerprint,
        )

    def test_stuck_recovery_holds_s_for_two_seconds(self) -> None:
        task = self.make_task()

        task.recover_from_stuck_entrance("佩里")

        task.send_key.assert_called_once_with(
            "s",
            down_time=2.0,
            after_sleep=1.0,
        )

    def test_boss_cycle_fights_then_leaves_room(self) -> None:
        task = self.make_task()
        exit_confirm_box = object()
        task.wait_for_battle_completion = MagicMock(
            return_value=exit_confirm_box
        )
        task.wait_until = MagicMock(return_value=True)
        scheduled = datetime(
            2026, 8, 10, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )

        result = task.run_boss_cycle("腐化根獸", scheduled)

        self.assertTrue(result)
        task.sleep.assert_any_call(2.0)
        self.assertEqual(
            task.send_key.call_args_list[0].kwargs,
            {"down_time": 0.3, "after_sleep": 1.5},
        )
        self.assertEqual(
            [call.args[0] for call in task.send_key.call_args_list],
            ["space"],
        )
        task.wait_for_battle_completion.assert_called_once_with()
        task.leave_boss_room.assert_called_once_with("腐化根獸")
        task.wait_until_boss_ready.assert_called_once_with(
            scheduled,
            "腐化根獸",
        )

    def test_boss_cycle_stops_if_exit_task_marker_times_out(self) -> None:
        task = self.make_task()
        task.wait_for_battle_completion = MagicMock(return_value=None)
        task.wait_until = MagicMock(return_value=True)
        scheduled = datetime(
            2026, 8, 10, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )

        result = task.run_boss_cycle("克拉瑪", scheduled)

        self.assertFalse(result)
        self.assertEqual(
            [call.args[0] for call in task.send_key.call_args_list],
            ["space"],
        )
        task.wait_for_battle_completion.assert_called_once_with()

    def test_leave_retries_when_minimap_does_not_change(self) -> None:
        task = self.make_task()
        task.wait_until = MagicMock(side_effect=[False, True])
        task._last_exit_minimap_difference = 8.5

        result = AutoWildBossTask.leave_boss_room(task, "克拉瑪")

        self.assertTrue(result)
        self.assertEqual(task.send_key.call_count, 2)
        task.open_exit_confirmation_for_retry.assert_called_once_with()
        self.assertEqual(
            task.wait_until.call_args_list[0].kwargs,
            {
                "time_out": task.EXIT_ATTEMPT_TIMEOUT_SECONDS,
                "settle_time": 1.0,
                "raise_if_not_found": False,
            },
        )

    def test_leave_stops_after_three_unchanged_minimap_attempts(self) -> None:
        task = self.make_task()
        task.wait_until = MagicMock(return_value=False)

        result = AutoWildBossTask.leave_boss_room(task, "克拉瑪")

        self.assertFalse(result)
        self.assertEqual(task.send_key.call_count, 3)
        self.assertEqual(
            task.open_exit_confirmation_for_retry.call_count,
            2,
        )

    def test_exit_retry_recaptures_room_and_reopens_confirmation(self) -> None:
        task = self.make_task()
        exit_task = object()
        exit_confirm = object()
        task.wait_for_feature = MagicMock(
            side_effect=[exit_task, exit_confirm]
        )
        task.find_one = MagicMock(return_value=None)
        task.move_and_click = MagicMock()

        result = AutoWildBossTask.open_exit_confirmation_for_retry(task)

        self.assertTrue(result)
        self.assertEqual(
            [call.args[0] for call in task.wait_for_feature.call_args_list],
            [task.EXIT_TASK_FEATURE, task.EXIT_CONFIRM_FEATURE],
        )
        task.capture_boss_room_exit_fingerprint.assert_called_once_with()
        task.move_and_click.assert_called_once_with(
            exit_task,
            after_sleep=0.1,
        )

    def test_exit_retry_uses_50_02_when_domain_is_second_item(self) -> None:
        task = self.make_task()
        secondary_marker = object()
        exit_button = object()
        exit_confirm = object()
        task.find_one = MagicMock(
            side_effect=[None, secondary_marker]
        )
        task.wait_for_feature = MagicMock(
            side_effect=[exit_button, exit_confirm]
        )
        task.move_and_click = MagicMock()

        result = AutoWildBossTask.open_exit_confirmation_for_retry(task)

        self.assertTrue(result)
        self.assertEqual(
            [call.args[0] for call in task.wait_for_feature.call_args_list],
            [
                task.SECONDARY_EXIT_BUTTON_FEATURE,
                task.EXIT_CONFIRM_FEATURE,
            ],
        )
        task.move_and_click.assert_called_once_with(
            exit_button,
            after_sleep=0.1,
        )

    def test_exit_retry_reuses_confirmation_that_is_still_open(self) -> None:
        task = self.make_task()
        task.find_one = MagicMock(return_value=object())
        task.wait_for_feature = MagicMock()

        result = AutoWildBossTask.open_exit_confirmation_for_retry(task)

        self.assertTrue(result)
        task.wait_for_feature.assert_not_called()
        task.capture_boss_room_exit_fingerprint.assert_not_called()

    def test_exit_fingerprint_is_copied_before_opening_dialog(self) -> None:
        task = self.make_task()
        fingerprint = np.arange(16, dtype=np.uint8)
        task.get_minimap_inner_fingerprint = MagicMock(
            return_value=fingerprint
        )

        result = AutoWildBossTask.capture_boss_room_exit_fingerprint(task)
        fingerprint[:] = 0

        self.assertTrue(result)
        np.testing.assert_array_equal(
            task._boss_room_exit_fingerprint,
            np.arange(16, dtype=np.uint8),
        )

    def test_outside_requires_minimap_change_and_room_markers_gone(self) -> None:
        task = self.make_task()
        task.is_main_screen = MagicMock(return_value=True)
        task._boss_room_exit_fingerprint = np.zeros(16, dtype=np.uint8)
        unchanged = np.zeros(16, dtype=np.uint8)
        changed = np.full(16, 10, dtype=np.uint8)
        task.get_minimap_inner_fingerprint = MagicMock(
            side_effect=[unchanged, changed]
        )
        task.find_any_feature = MagicMock(return_value=None)
        task.find_one = MagicMock(return_value=None)

        self.assertFalse(AutoWildBossTask.is_outside_boss_room(task))
        self.assertTrue(AutoWildBossTask.is_outside_boss_room(task))
        task.find_any_feature.assert_called_once_with(
            task.ROOM_READY_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            [task.EXIT_TASK_FEATURE, task.SECONDARY_EXIT_TASK_FEATURE],
        )

    def test_battle_monitor_moves_mouse_away_after_skipping_cutscene(self) -> None:
        task = self.make_task()
        skip_box = object()
        task.find_one = MagicMock(side_effect=[None, skip_box])
        task.find_any_feature = MagicMock(return_value=None)
        task.move_and_click = MagicMock()

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(result)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            [
                task.EXIT_CONFIRM_FEATURE,
                task.SKIP_CUTSCENE_FEATURE,
            ],
        )
        task.find_any_feature.assert_called_once_with(
            task.VICTORY_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )
        task.move_and_click.assert_called_once_with(
            skip_box,
            after_sleep=0.1,
        )
        task.pydirect_interaction.move.assert_called_once_with(800, 450)
        task.sleep.assert_called_once_with(0.9)

    def test_battle_monitor_confirms_victory_and_keeps_waiting(self) -> None:
        task = self.make_task()
        victory_box = object()
        task.find_one = MagicMock(return_value=None)
        task.find_any_feature = MagicMock(return_value=victory_box)
        task.move_and_click = MagicMock()

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(result)
        task.find_any_feature.assert_called_once_with(
            task.VICTORY_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )
        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=1.0,
        )
        task.move_and_click.assert_not_called()

    def test_battle_monitor_clicks_exit_task_then_keeps_watching(self) -> None:
        task = self.make_task()
        task._battle_victory_confirmed = True
        exit_task_box = object()
        task.find_one = MagicMock(
            side_effect=[None, None, exit_task_box]
        )
        task.find_any_feature = MagicMock(return_value=None)
        task.move_and_click = MagicMock()

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(result)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            [
                task.EXIT_CONFIRM_FEATURE,
                task.SKIP_CUTSCENE_FEATURE,
                task.EXIT_TASK_FEATURE,
            ],
        )
        task.move_and_click.assert_called_once_with(
            exit_task_box,
            after_sleep=0.1,
        )
        task.capture_boss_room_exit_fingerprint.assert_called_once_with()
        task.pydirect_interaction.move.assert_called_once_with(800, 450)
        task.sleep.assert_called_once_with(0.9)

    def test_battle_monitor_clicks_50_02_for_second_domain_item(self) -> None:
        task = self.make_task()
        task._battle_victory_confirmed = True
        secondary_marker = object()
        exit_button = object()
        task.find_one = MagicMock(
            side_effect=[
                None,
                None,
                None,
                secondary_marker,
                exit_button,
            ]
        )
        task.find_any_feature = MagicMock(return_value=None)
        task.move_and_click = MagicMock()

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(result)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            [
                task.EXIT_CONFIRM_FEATURE,
                task.SKIP_CUTSCENE_FEATURE,
                task.EXIT_TASK_FEATURE,
                task.SECONDARY_EXIT_TASK_FEATURE,
                task.SECONDARY_EXIT_BUTTON_FEATURE,
            ],
        )
        task.move_and_click.assert_called_once_with(
            exit_button,
            after_sleep=0.1,
        )

    def test_battle_monitor_finishes_on_exit_confirmation(self) -> None:
        task = self.make_task()
        exit_confirm_box = object()
        task.find_one = MagicMock(return_value=exit_confirm_box)

        result = task.find_battle_completion_or_handle_overlay()

        self.assertIs(result, exit_confirm_box)
        task.find_one.assert_called_once_with(
            task.EXIT_CONFIRM_FEATURE,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
        )

    def test_exit_task_before_victory_does_not_finish_battle(self) -> None:
        task = self.make_task()
        exit_task_box = object()
        victory_box = object()
        task.find_one = MagicMock(
            side_effect=[
                None,
                None,
                exit_task_box,
                None,
                None,
                None,
                exit_task_box,
            ]
        )
        task.find_any_feature = MagicMock(
            side_effect=[None, victory_box, None]
        )
        task.move_and_click = MagicMock()

        before_victory = task.find_battle_completion_or_handle_overlay()
        victory = task.find_battle_completion_or_handle_overlay()
        after_victory = task.find_battle_completion_or_handle_overlay()

        self.assertIsNone(before_victory)
        self.assertIsNone(victory)
        self.assertIsNone(after_victory)
        self.assertTrue(task._battle_victory_confirmed)
        task.move_and_click.assert_called_once_with(
            exit_task_box,
            after_sleep=0.1,
        )
        task.pydirect_interaction.move.assert_called_once_with(800, 450)
        task.send_key.assert_called_once_with(
            "space",
            down_time=0.08,
            after_sleep=1.0,
        )

    def test_boss_readiness_is_not_checked_before_scheduled_plus_10(self) -> None:
        task = self.make_task()
        scheduled = datetime(
            2026, 8, 10, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )
        task.current_hong_kong_time = MagicMock(
            side_effect=[
                scheduled,
                scheduled + timedelta(seconds=10),
            ]
        )
        task.is_boss_ready_to_fight = MagicMock(return_value=True)

        result = AutoWildBossTask.wait_until_boss_ready(
            task,
            scheduled,
            "克拉瑪",
        )

        self.assertTrue(result)
        task.sleep.assert_called_once_with(1)
        task.is_boss_ready_to_fight.assert_called_once_with()

    def test_waiting_screen_is_not_ready_to_fight(self) -> None:
        task = self.make_task()
        task.is_main_screen = MagicMock(return_value=True)
        task.find_all_features = MagicMock(return_value=object())
        task.find_room_ready_feature_with_scores = MagicMock()

        self.assertFalse(task.is_boss_ready_to_fight())
        task.find_all_features.assert_called_once_with(
            task.WAITING_FOR_BOSS_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.8,
        )
        task.find_room_ready_feature_with_scores.assert_not_called()

    def test_ready_requires_waiting_gone_and_room_ready_present(self) -> None:
        task = self.make_task()
        task.is_main_screen = MagicMock(return_value=True)
        task.find_all_features = MagicMock(return_value=None)
        task.find_room_ready_feature_with_scores = MagicMock(
            return_value=object()
        )

        self.assertTrue(task.is_boss_ready_to_fight())
        task.find_all_features.assert_called_once_with(
            task.WAITING_FOR_BOSS_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.8,
        )
        task.find_room_ready_feature_with_scores.assert_called_once_with()

    def test_room_ready_uses_color_scores_without_mutating_templates(self) -> None:
        task = self.make_task()
        low = SimpleNamespace(confidence=0.42)
        matched = SimpleNamespace(confidence=0.85)
        task.find_one = MagicMock(side_effect=[low, matched, None])

        result = task.find_room_ready_feature_with_scores()

        self.assertIs(result, matched)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            ["20_01", "33_01", "34_01"],
        )
        for find_call in task.find_one.call_args_list:
            self.assertEqual(
                find_call.kwargs,
                {
                    "horizontal_variance": 0.04,
                    "vertical_variance": 0.04,
                    "threshold": -1.0,
                    "use_gray_scale": False,
                },
            )
        task.log_info.assert_called_once_with(
            "房間就緒辨識分數：20_01=0.420、33_01=0.850、"
            "34_01=0.000；門檻=0.82"
        )

    def test_bottom_confirm_refreshes_full_room_ready_window(self) -> None:
        task = self.make_task()
        scheduled = datetime(
            2026, 8, 10, 12, 0, tzinfo=HONG_KONG_TIMEZONE
        )
        first_check = datetime(
            2026, 8, 10, 12, 10, tzinfo=HONG_KONG_TIMEZONE
        )
        task._bottom_confirm_last_handled_at = float("-inf")

        def readiness_side_effect():
            if task.is_boss_ready_to_fight.call_count == 1:
                task._bottom_confirm_last_handled_at = 5.0
                return False
            return True

        task.is_boss_ready_to_fight = MagicMock(
            side_effect=readiness_side_effect
        )
        task.current_hong_kong_time = MagicMock(
            side_effect=[
                first_check,
                first_check + timedelta(seconds=1),
                first_check + timedelta(seconds=10.5),
            ]
        )

        result = AutoWildBossTask.wait_until_boss_ready(
            task,
            scheduled,
            "test boss",
        )

        self.assertTrue(result)
        self.assertEqual(task.is_boss_ready_to_fight.call_count, 2)
        task.log_info.assert_any_call(
            "已關閉 37_01；重新給 test boss 完整 10 秒房間就緒辨識時間"
        )

    def test_find_all_features_requires_every_feature(self) -> None:
        task = self.make_task()
        first = object()
        task.find_one = MagicMock(side_effect=[first, object()])

        result = task.find_all_features(("31_01", "31_02"), threshold=0.8)

        self.assertIs(result, first)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            ["31_01", "31_02"],
        )

        task.find_one = MagicMock(side_effect=[first, None])
        self.assertIsNone(
            task.find_all_features(("31_01", "31_02"), threshold=0.8)
        )

    def test_find_any_feature_accepts_one_match(self) -> None:
        task = self.make_task()
        match = object()
        task.find_one = MagicMock(side_effect=[None, match, object()])

        result = task.find_any_feature(
            ("20_01", "33_01", "34_01"),
            threshold=0.82,
        )

        self.assertIs(result, match)
        self.assertEqual(
            [call.args[0] for call in task.find_one.call_args_list],
            ["20_01", "33_01"],
        )

    def test_boss_feature_mapping_uses_numbered_templates(self) -> None:
        self.assertEqual(AutoWildBossTask.MENU_FEATURE, "28_01")
        self.assertEqual(
            AutoWildBossTask.MOVE_CONFIRM_FEATURES,
            ("30_01", "30_02"),
        )
        self.assertEqual(
            AutoWildBossTask.ENTRY_CONFIRM_FEATURES,
            ("16_01", "16_02"),
        )
        self.assertEqual(
            AutoWildBossTask.ENTRY_NOT_READY_FEATURES,
            ("29_01", "29_02"),
        )
        self.assertEqual(
            AutoWildBossTask.WAITING_FOR_BOSS_FEATURES,
            ("31_01", "31_02"),
        )
        self.assertEqual(
            AutoWildBossTask.ROOM_READY_FEATURES,
            ("20_01", "33_01", "34_01"),
        )
        self.assertEqual(
            AutoWildBossTask.VICTORY_FEATURES,
            ("36_01", "17_01"),
        )
        self.assertEqual(AutoWildBossTask.SKIP_CUTSCENE_FEATURE, "35_01")
        self.assertEqual(AutoWildBossTask.EXIT_TASK_FEATURE, "18_01")
        self.assertEqual(AutoWildBossTask.EXIT_CONFIRM_FEATURE, "19_01")
        self.assertEqual(
            AutoWildBossTask.SECONDARY_EXIT_TASK_FEATURE,
            "50_01",
        )
        self.assertEqual(
            AutoWildBossTask.SECONDARY_EXIT_BUTTON_FEATURE,
            "50_02",
        )

    def test_exit_task_click_uses_18_01_center(self) -> None:
        task = self.make_task()
        box = SimpleNamespace(
            x=1498,
            y=253,
            width=81,
            height=28,
            name="18_01",
        )

        task.move_and_click(box, after_sleep=0.1)

        self.assertEqual(
            task.pydirect_interaction.move.call_args_list,
            [call(1538, 267), call(800, 450)],
        )
        task.pydirect_interaction.click.assert_called_once_with(
            down_time=0.1
        )
        task.check_bottom_confirm_before_action.assert_called_once_with()
        self.assertEqual(task.sleep.call_args_list, [call(0.2), call(1.0)])

    def test_boss_row_scroll_and_click_each_check_bottom_confirm(self) -> None:
        task = self.make_task()
        list_index = len(task.BOSS_ROW_Y)

        task.select_boss_row(list_index, "test boss")

        task.pydirect_interaction.scroll.assert_called_once()
        task.pydirect_interaction.click.assert_called_once_with(
            down_time=0.1
        )
        self.assertEqual(
            task.pydirect_interaction.move.call_args_list[-1],
            call(800, 450),
        )
        self.assertEqual(
            task.sleep.call_args_list[-2:],
            [call(1.0), call(0.5)],
        )
        self.assertEqual(
            task.check_bottom_confirm_before_action.call_count,
            2,
        )

    def test_navigate_uses_space_immediately_after_boss_selection(self) -> None:
        task = self.make_task()
        task.focus_game_window = MagicMock()
        task.ensure_main_screen = MagicMock(return_value=True)
        menu_box = object()
        move_confirm_box = object()
        task.wait_until = MagicMock(
            side_effect=[menu_box, move_confirm_box]
        )
        task.find_one = MagicMock(return_value=move_confirm_box)
        task.move_and_click = MagicMock()
        task.select_boss_row = MagicMock()

        result = AutoWildBossTask.navigate_to_boss(
            task,
            2,
            "克拉瑪",
        )

        self.assertTrue(result)
        task.select_boss_row.assert_called_once_with(2, "克拉瑪")
        self.assertEqual(
            [call.args[0] for call in task.send_key.call_args_list],
            ["esc", "space", "space"],
        )
        self.assertEqual(
            task.send_key.call_args_list[-1].kwargs,
            {"down_time": 0.1, "after_sleep": 3.0},
        )

    def test_navigate_stops_if_move_confirm_does_not_appear(self) -> None:
        task = self.make_task()
        task.focus_game_window = MagicMock()
        task.ensure_main_screen = MagicMock(return_value=True)
        task.wait_until = MagicMock(side_effect=[object(), None])
        task.move_and_click = MagicMock()
        task.select_boss_row = MagicMock()

        with self.assertRaisesRegex(RuntimeError, r"30_01 \+ 30_02"):
            AutoWildBossTask.navigate_to_boss(task, 0, "佩里")

        self.assertEqual(
            [call.args[0] for call in task.send_key.call_args_list],
            ["esc", "space"],
        )

if __name__ == "__main__":
    unittest.main()

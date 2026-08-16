import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

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
            ["space", "space"],
        )
        task.wait_for_battle_completion.assert_called_once_with()
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
        task.pydirect_interaction.move.assert_called_once_with(800, 450)
        task.sleep.assert_called_once_with(0.9)

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
        task.find_any_feature = MagicMock()

        self.assertFalse(task.is_boss_ready_to_fight())
        task.find_all_features.assert_called_once_with(
            task.WAITING_FOR_BOSS_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.8,
        )
        task.find_any_feature.assert_not_called()

    def test_ready_requires_waiting_gone_and_room_ready_present(self) -> None:
        task = self.make_task()
        task.is_main_screen = MagicMock(return_value=True)
        task.find_all_features = MagicMock(return_value=None)
        task.find_any_feature = MagicMock(return_value=object())

        self.assertTrue(task.is_boss_ready_to_fight())
        task.find_all_features.assert_called_once_with(
            task.WAITING_FOR_BOSS_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.8,
        )
        task.find_any_feature.assert_called_once_with(
            task.ROOM_READY_FEATURES,
            horizontal_variance=0.04,
            vertical_variance=0.04,
            threshold=0.82,
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

        task.pydirect_interaction.move.assert_called_once_with(1538, 267)
        task.pydirect_interaction.click.assert_called_once_with(
            down_time=0.1
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

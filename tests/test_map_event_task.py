import unittest
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ok import Box
from ok.feature.FeatureSet import FeatureSet

from src.plugins.map_events.tasks.AutoMapEventTask import (
    AutoMapEventTask,
    DEEP_HOLE_EVENT,
    EVENT_PRIORITY,
    OMINOUS_EVENT,
    TARGET_REGION_CONFIG_KEY,
    TARGET_REGION_FEATURES,
    TARGET_REGION_LOCAL_FEATURES,
    UNDERGROUND_HOLE_EVENT,
    choose_available_event,
    event_detail_panel_is_visible,
    event_row_is_available,
    seconds_until_next_minute_five,
    selected_events_from_config,
)


class MapEventTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.feature_set = FeatureSet(
            False,
            str(cls.project_root / "assets" / "coco_annotations.json"),
            0.02,
            0.03,
            0.8,
        )

    def load_asset_frame(self, number: str):
        frame = cv2.imread(
            str(self.project_root / "assets" / "images" / f"{number}.png")
        )
        self.assertIsNotNone(frame)
        return frame

    @staticmethod
    def synthetic_row_frame(coloured_event_ids=()):
        frame = np.full((900, 1600, 3), 100, dtype=np.uint8)
        rows = {
            OMINOUS_EVENT.event_id: [Box(82, 744, 110, 16, 1.0, "46_01")],
            DEEP_HOLE_EVENT.event_id: [Box(82, 688, 141, 16, 1.0, "46_02")],
            UNDERGROUND_HOLE_EVENT.event_id: [
                Box(82, 632, 141, 16, 1.0, "46_03")
            ],
        }
        for event_id, event_rows in rows.items():
            if event_id not in coloured_event_ids:
                continue
            row = event_rows[0]
            frame[row.y - 25 : row.y + 35, row.x - 62 : row.x - 4] = (
                255,
                0,
                255,
            )
        return frame, rows

    def test_gui_selection_keeps_required_priority(self) -> None:
        selected = selected_events_from_config(
            {
                OMINOUS_EVENT.config_key: True,
                DEEP_HOLE_EVENT.config_key: True,
                UNDERGROUND_HOLE_EVENT.config_key: True,
            }
        )
        self.assertEqual(selected, EVENT_PRIORITY)
        self.assertEqual(
            [event.display_name for event in selected],
            [
                "不祥的召喚結界",
                "通往深層的黑色坑洞",
                "通往地下的黑色坑洞",
            ],
        )

    def test_only_configured_hunting_region_is_frost_canyon(self) -> None:
        self.assertEqual(TARGET_REGION_CONFIG_KEY, "指定狩獵場地區")
        self.assertEqual(TARGET_REGION_FEATURES, {"冰霜狹谷": "56_01"})
        self.assertEqual(TARGET_REGION_LOCAL_FEATURES, {"冰霜狹谷": "56_02"})

    def test_world_map_templates_match_reference_frames(self) -> None:
        cases = (("0", "55_01"), ("1", "56_01"))
        for asset, feature in cases:
            with self.subTest(asset=asset, feature=feature):
                matches = self.feature_set.find_one_feature(
                    self.load_asset_frame(asset),
                    feature,
                    threshold=0.85,
                    limit=1,
                )
                self.assertTrue(matches)

    def test_local_region_template_distinguishes_frost_canyon(self) -> None:
        frost_map = self.load_asset_frame("0")
        other_map = self.load_asset_frame("1")
        self.assertTrue(
            self.feature_set.find_one_feature(
                frost_map,
                "56_02",
                threshold=0.85,
                limit=1,
            )
        )
        self.assertFalse(
            self.feature_set.find_one_feature(
                other_map,
                "56_02",
                threshold=0.85,
                limit=1,
            )
        )

    def test_grey_ominous_row_is_not_available(self) -> None:
        frame, rows = self.synthetic_row_frame()
        row = rows[OMINOUS_EVENT.event_id][0]
        self.assertFalse(event_row_is_available(frame, row))

    def test_coloured_ominous_row_is_available(self) -> None:
        frame, rows = self.synthetic_row_frame((OMINOUS_EVENT.event_id,))
        row = rows[OMINOUS_EVENT.event_id][0]
        self.assertTrue(event_row_is_available(frame, row))

    def test_priority_uses_deep_hole_when_ominous_is_grey(self) -> None:
        frame, rows = self.synthetic_row_frame(
            (DEEP_HOLE_EVENT.event_id, UNDERGROUND_HOLE_EVENT.event_id)
        )
        choice = choose_available_event(EVENT_PRIORITY, rows, frame)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[0], DEEP_HOLE_EVENT)

    def test_priority_uses_ominous_when_all_are_available(self) -> None:
        frame, rows = self.synthetic_row_frame(
            tuple(event.event_id for event in EVENT_PRIORITY)
        )
        choice = choose_available_event(EVENT_PRIORITY, rows, frame)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[0], OMINOUS_EVENT)

    def test_event_detail_panel_colour_marker(self) -> None:
        frame = np.zeros((900, 1600, 3), dtype=np.uint8)
        self.assertFalse(event_detail_panel_is_visible(frame))
        frame[round(900 * 0.87) : round(900 * 0.97), round(1600 * 0.55) : round(1600 * 0.96)] = (
            255,
            90,
            126,
        )
        self.assertTrue(event_detail_panel_is_visible(frame))

    def test_zero_and_one_templates_are_separate(self) -> None:
        cases = (
            ("1", "48_01", "44_01"),
            ("0", "44_01", "48_01"),
            ("0", "49_01", "51_01"),
            ("1", "51_01", "49_01"),
        )
        for asset, expected, rejected in cases:
            with self.subTest(asset=asset, expected=expected):
                frame = self.load_asset_frame(asset)
                exact = self.feature_set.find_one_feature(
                    frame,
                    expected,
                    horizontal_variance=0.04,
                    vertical_variance=0.04,
                    threshold=0.85,
                    limit=1,
                )
                mismatch = self.feature_set.find_one_feature(
                    frame,
                    rejected,
                    horizontal_variance=0.04,
                    vertical_variance=0.04,
                    threshold=0.85,
                    limit=1,
                )
                self.assertTrue(exact)
                self.assertFalse(mismatch)

    def test_black_hole_stage_templates_match_native_path(self) -> None:
        for shot, feature in (
            ("5", "41_01"),
            ("1", "42_01"),
            ("2", "40_01"),
        ):
            with self.subTest(shot=shot):
                matches = self.feature_set.find_one_feature(
                    self.load_asset_frame(shot),
                    feature,
                    horizontal_variance=0.04,
                    vertical_variance=0.04,
                    threshold=0.85,
                    limit=1,
                )
                self.assertTrue(matches)
                self.assertGreaterEqual(matches[0].confidence, 0.99)

    def test_room_ready_and_complete_stages_do_not_cross_match(self) -> None:
        cases = (
            ("2", "42_01"),
            ("1", "40_01"),
        )
        for asset, rejected_feature in cases:
            with self.subTest(asset=asset, rejected_feature=rejected_feature):
                matches = self.feature_set.find_one_feature(
                    self.load_asset_frame(asset),
                    rejected_feature,
                    threshold=0.85,
                    limit=1,
                )
                self.assertFalse(matches)

    def test_wait_to_minute_five(self) -> None:
        self.assertEqual(
            seconds_until_next_minute_five(datetime(2026, 8, 19, 10, 3, 30)),
            90.0,
        )
        self.assertEqual(
            seconds_until_next_minute_five(datetime(2026, 8, 19, 10, 5, 0)),
            3600.0,
        )

    def test_count_state_uses_zero_only_for_initial_check(self) -> None:
        class Harness:
            STATE_MATCH_THRESHOLD = 0.85

            def __init__(self, matches):
                self.matches = set(matches)

            def find_one(self, feature, **_kwargs):
                return object() if feature in self.matches else None

            def sleep(self, _seconds):
                return None

        zero = Harness((OMINOUS_EVENT.zero_feature,))
        one = Harness((OMINOUS_EVENT.one_feature,))
        many = Harness(())
        self.assertEqual(
            AutoMapEventTask.detect_count_state(
                zero, OMINOUS_EVENT, check_zero=True
            ),
            "zero",
        )
        self.assertEqual(
            AutoMapEventTask.detect_count_state(
                zero, OMINOUS_EVENT, check_zero=False
            ),
            "many",
        )
        self.assertEqual(
            AutoMapEventTask.detect_count_state(
                one, OMINOUS_EVENT, check_zero=True
            ),
            "one",
        )
        self.assertEqual(
            AutoMapEventTask.detect_count_state(
                many, OMINOUS_EVENT, check_zero=True
            ),
            "many",
        )

    def test_black_hole_flow_uses_required_space_sequence(self) -> None:
        class Harness:
            BLACK_HOLE_ENTRY_FEATURE = "41_01"
            BLACK_HOLE_ROOM_READY_FEATURE = "42_01"
            BLACK_HOLE_COMPLETE_FEATURE = "40_01"
            TRAVEL_TIMEOUT_SECONDS = 300
            ROOM_LOAD_TIMEOUT_SECONDS = 120
            BATTLE_TIMEOUT_SECONDS = 900
            BLACK_HOLE_POST_BATTLE_SECONDS = 20

            def __init__(self):
                self.actions = []

            def log_info(self, _message):
                return None

            def wait_for_stage(self, feature, timeout, _message):
                self.actions.append(("wait", feature, timeout))
                return object()

            def send_key(self, key, **_kwargs):
                self.actions.append(("key", key))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

        harness = Harness()
        AutoMapEventTask.complete_black_hole_event(harness, DEEP_HOLE_EVENT)
        self.assertEqual(
            harness.actions,
            [
                ("wait", "41_01", 300),
                ("key", "space"),
                ("wait", "42_01", 120),
                ("key", "space"),
                ("wait", "40_01", 900),
                ("sleep", 20),
            ],
        )

    def test_event_row_click_uses_real_mouse_input(self) -> None:
        class Interaction:
            def __init__(self, actions):
                self.actions = actions

            def move(self, x, y):
                self.actions.append(("move", x, y))

            def click(self, **kwargs):
                self.actions.append(("click", kwargs))

        class Harness:
            def __init__(self):
                self.actions = []
                self.pydirect_interaction = Interaction(self.actions)
                self.width = 1600
                self.height = 900

            def log_info(self, _message):
                return None

            def ensure_in_front(self):
                self.actions.append(("foreground",))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

            def check_bottom_confirm_before_action(self):
                self.actions.append(("popup_check",))

        harness = Harness()
        row = Box(82, 688, 141, 16, 1.0, "46_02")
        AutoMapEventTask.click_event_row(harness, row)
        self.assertEqual(
            harness.actions,
            [
                ("foreground",),
                ("move", 152, 696),
                ("sleep", 0.2),
                ("popup_check",),
                ("click", {"down_time": 0.1}),
                ("sleep", 1.0),
                ("move", 800, 450),
            ],
        )

    def test_region_click_uses_top_centre_then_returns_mouse(self) -> None:
        class Interaction:
            def __init__(self, actions):
                self.actions = actions

            def move(self, x, y):
                self.actions.append(("move", x, y))

            def click(self, **kwargs):
                self.actions.append(("click", kwargs))

        class Harness:
            CLICK_RETURN_DELAY_SECONDS = 1.0
            width = 1600
            height = 900

            def __init__(self):
                self.actions = []
                self.pydirect_interaction = Interaction(self.actions)

            def log_info(self, _message):
                return None

            def ensure_in_front(self):
                self.actions.append(("foreground",))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

            def check_bottom_confirm_before_action(self):
                self.actions.append(("popup_check",))

        harness = Harness()
        region = Box(658, 483, 108, 26, 1.0, "56_01")
        AutoMapEventTask.click_map_box(
            harness,
            region,
            click_at_top=True,
        )
        self.assertEqual(
            harness.actions,
            [
                ("foreground",),
                ("move", 712, 483),
                ("sleep", 0.2),
                ("popup_check",),
                ("click", {"down_time": 0.1}),
                ("sleep", 1.0),
                ("move", 800, 450),
            ],
        )

    def test_world_map_entry_waits_total_one_point_five_seconds(self) -> None:
        class Interaction:
            def __init__(self, actions):
                self.actions = actions

            def move(self, x, y):
                self.actions.append(("move", x, y))

            def click(self, **kwargs):
                self.actions.append(("click", kwargs))

        class Harness:
            CLICK_RETURN_DELAY_SECONDS = 1.0
            width = 1600
            height = 900

            def __init__(self):
                self.actions = []
                self.pydirect_interaction = Interaction(self.actions)

            def log_info(self, _message):
                return None

            def ensure_in_front(self):
                return None

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

            def check_bottom_confirm_before_action(self):
                return None

        harness = Harness()
        entry = Box(74, 37, 93, 23, 1.0, "55_01")
        AutoMapEventTask.click_map_box(
            harness,
            entry,
            click_at_top=False,
            total_after_click=1.5,
        )
        self.assertEqual(
            [action for action in harness.actions if action[0] == "sleep"],
            [("sleep", 0.2), ("sleep", 1.0), ("sleep", 0.5)],
        )

    def test_world_map_drag_holds_mouse_and_moves_in_visible_steps(self) -> None:
        class Interaction:
            def __init__(self, actions):
                self.actions = actions

            def move(self, x, y):
                self.actions.append(("move", x, y))

            def mouse_down(self, **kwargs):
                self.actions.append(("mouse_down", kwargs))

            def mouse_up(self, **kwargs):
                self.actions.append(("mouse_up", kwargs))

        class Harness:
            width = 1600
            height = 900
            WORLD_MAP_DRAG_STEPS = 3
            WORLD_MAP_DRAG_DURATION_SECONDS = 0.3
            WORLD_MAP_DRAG_SETTLE_SECONDS = 0.4

            def __init__(self):
                self.actions = []
                self.pydirect_interaction = Interaction(self.actions)

            def ensure_in_front(self):
                self.actions.append(("foreground",))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

            def check_bottom_confirm_before_action(self):
                self.actions.append(("popup_check",))

        harness = Harness()
        AutoMapEventTask.drag_world_map(harness, 0.50, 0.44, 0.78, 0.44)
        self.assertEqual(harness.actions[0], ("foreground",))
        self.assertEqual(harness.actions[1], ("move", 800, 396))
        self.assertEqual(harness.actions[4], ("mouse_down", {"key": "left"}))
        self.assertEqual(
            [action for action in harness.actions if action[0] == "move"][-1],
            ("move", 1248, 396),
        )
        self.assertEqual(
            [action for action in harness.actions if action[0] == "mouse_up"],
            [("mouse_up", {"key": "left"})],
        )
        self.assertEqual(harness.actions[-1], ("sleep", 0.4))

    def test_world_map_search_uses_requested_view_direction_order(self) -> None:
        self.assertEqual(
            AutoMapEventTask.WORLD_MAP_DRAG_DIRECTIONS,
            (
                (0.82, 0.45, 0.18, 0.45),  # view right
                (0.50, 0.22, 0.50, 0.70),  # view up
                (0.50, 0.70, 0.50, 0.22),  # view down
                (0.18, 0.45, 0.82, 0.45),  # view left
                (0.50, 0.22, 0.50, 0.70),  # view up
                (0.50, 0.70, 0.50, 0.22),  # view down
            ),
        )

    def test_open_map_selects_world_map_and_region_before_event_list(self) -> None:
        class Harness:
            WORLD_MAP_ENTRY_FEATURE = "55_01"
            WORLD_MAP_ENTRY_THRESHOLD = 0.85
            TARGET_REGION_THRESHOLD = 0.85
            MAP_OPEN_TIMEOUT_SECONDS = 10
            WORLD_MAP_OPEN_DELAY_SECONDS = 1.5
            config = {TARGET_REGION_CONFIG_KEY: "冰霜狹谷"}

            def __init__(self):
                self.actions = []
                self.entry = Box(74, 37, 93, 23, 1.0, "55_01")
                self.region = Box(658, 483, 108, 26, 1.0, "56_01")

            def log_info(self, message):
                self.actions.append(("log", message))

            def send_key(self, key, **kwargs):
                self.actions.append(("key", key, kwargs))

            def _world_map_control_search_box(self):
                return Box(32, 9, 320, 90)

            def wait_for_feature_in_box(self, feature, box, threshold, timeout):
                self.actions.append(("wait_feature", feature, threshold, timeout))
                return self.entry

            def find_current_target_region(self, feature):
                self.actions.append(("find_current_region", feature))
                return None

            def click_map_box(self, box, **kwargs):
                self.actions.append(("click_box", box.name, kwargs))

            def find_target_region_with_drag(self, feature):
                self.actions.append(("find_region", feature))
                return self.region

            def wait_for_map_event_list(self):
                self.actions.append(("confirm_event_list",))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

        harness = Harness()
        AutoMapEventTask.open_map(harness)
        action_names = [action[0] for action in harness.actions]
        self.assertEqual(
            action_names,
            [
                "log",
                "key",
                "find_current_region",
                "wait_feature",
                "log",
                "click_box",
                "find_region",
                "log",
                "click_box",
                "confirm_event_list",
            ],
        )
        self.assertEqual(
            harness.actions[5],
            (
                "click_box",
                "55_01",
                {"click_at_top": False, "total_after_click": 1.5},
            ),
        )
        self.assertEqual(harness.actions[6], ("find_region", "56_01"))
        self.assertEqual(
            harness.actions[8],
            ("click_box", "56_01", {"click_at_top": True}),
        )

    def test_open_map_skips_world_map_when_already_in_target_region(self) -> None:
        class Harness:
            MAP_OPEN_TIMEOUT_SECONDS = 10
            config = {TARGET_REGION_CONFIG_KEY: "冰霜狹谷"}

            def __init__(self):
                self.actions = []

            def log_info(self, message):
                self.actions.append(("log", message))

            def send_key(self, key, **kwargs):
                self.actions.append(("key", key, kwargs))

            def find_current_target_region(self, feature):
                self.actions.append(("find_current_region", feature))
                return Box(97, 826, 78, 20, 1.0, feature)

            def wait_for_map_event_list(self):
                self.actions.append(("confirm_event_list",))

        harness = Harness()
        AutoMapEventTask.open_map(harness)
        self.assertEqual(
            [action[0] for action in harness.actions],
            ["log", "key", "find_current_region", "log", "confirm_event_list"],
        )
        self.assertEqual(harness.actions[2], ("find_current_region", "56_02"))

    def test_detected_event_focuses_game_and_waits_before_click_flow(self) -> None:
        class Harness:
            FOREGROUND_SETTLE_SECONDS = 2.0

            def __init__(self):
                self.actions = []

            def log_info(self, message):
                self.actions.append(("log", message))

            def focus_game_window(self):
                self.actions.append(("focus_game_window",))

            def sleep(self, seconds):
                self.actions.append(("sleep", seconds))

        harness = Harness()
        AutoMapEventTask.prepare_foreground_after_event_detection(
            harness
        )
        self.assertEqual(
            harness.actions,
            [
                (
                    "log",
                    "辨識到可進入的地圖事件，將遊戲視窗拉到前景並等待 2 秒",
                ),
                ("focus_game_window",),
                ("sleep", 2.0),
            ],
        )

    def test_startup_main_screen_check_matches_boss_recovery(self) -> None:
        class Harness:
            MAIN_SCREEN_FEATURE = "main_screen_marker_leftdown"

            def __init__(self):
                self.visible = False
                self.actions = []

            def find_one(self, feature, **kwargs):
                self.actions.append(("find", feature, kwargs))
                return object() if self.visible else None

            def is_main_screen(self):
                return AutoMapEventTask.is_main_screen(self)

            def ensure_in_front(self):
                self.actions.append(("foreground",))

            def send_key(self, key, **kwargs):
                self.actions.append(("key", key, kwargs))

            def wait_until(self, condition, **kwargs):
                self.actions.append(("wait", kwargs))
                self.visible = True
                return condition()

            def log_info(self, message):
                self.actions.append(("log", message))

        harness = Harness()
        self.assertTrue(AutoMapEventTask.ensure_main_screen(harness))
        self.assertEqual(
            [action[0] for action in harness.actions],
            ["find", "foreground", "key", "wait", "find", "log"],
        )
        key_action = harness.actions[2]
        self.assertEqual(key_action[1], "esc")
        self.assertEqual(key_action[2]["down_time"], 0.08)
        self.assertEqual(key_action[2]["after_sleep"], 0.8)

    def test_map_confirmation_accepts_any_event_row(self) -> None:
        class Harness:
            MAP_CONFIRM_FEATURES = ("46_01", "46_02", "46_03", "46_04")
            MAP_CONFIRM_THRESHOLD = 0.80
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)

            def _map_row_search_box(self):
                return Box(48, 522, 208, 261)

            def find_one(self, feature, **kwargs):
                self.calls.append((feature, kwargs))
                scores = {
                    "46_01": 0.73,
                    "46_02": 0.84,
                    "46_03": 0.97,
                    "46_04": 0.85,
                }
                return Box(82, 632, 141, 16, scores[feature], feature)

            def __init__(self):
                self.calls = []

        harness = Harness()
        match, scores = AutoMapEventTask.find_map_confirmation(harness)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "46_03")
        self.assertEqual(scores["46_01"], 0.73)
        self.assertEqual(len(harness.calls), 4)
        for _feature, kwargs in harness.calls:
            self.assertEqual(kwargs["threshold"], -1.0)
            self.assertEqual(kwargs["limit"], 1)

    def test_map_confirmation_rejects_all_weak_scores(self) -> None:
        class Harness:
            MAP_CONFIRM_FEATURES = ("46_01", "46_02", "46_03", "46_04")
            MAP_CONFIRM_THRESHOLD = 0.80
            frame = np.zeros((900, 1600, 3), dtype=np.uint8)

            def _map_row_search_box(self):
                return Box(48, 522, 208, 261)

            def find_one(self, feature, **_kwargs):
                return Box(82, 632, 141, 16, 0.79, feature)

        match, scores = AutoMapEventTask.find_map_confirmation(Harness())
        self.assertIsNone(match)
        self.assertEqual(set(scores), set(Harness.MAP_CONFIRM_FEATURES))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.plugins.wild_boss.data import (
    DEFAULT_BOSS_SCHEDULE,
    get_boss_definition,
)
from src.plugins.wild_boss.models import (
    BossScheduleItem,
    BossScheduleSettings,
    HONG_KONG_TIMEZONE,
    find_due_occurrences,
)
from src.plugins.wild_boss.storage import BossScheduleStorage


class WildBossScheduleTest(unittest.TestCase):
    def test_default_lead_time_is_five_minutes(self) -> None:
        self.assertEqual(DEFAULT_BOSS_SCHEDULE.lead_minutes, 5)

    def make_settings(self, time_hhmm: str) -> BossScheduleSettings:
        return BossScheduleSettings(
            items=[
                BossScheduleItem(True, time_hhmm, "clama"),
                BossScheduleItem(False, "06:00", "clama"),
                BossScheduleItem(False, "12:00", "peri"),
                BossScheduleItem(False, "18:00", "corrupted_root_beast"),
            ],
            lead_minutes=5,
            retry_seconds=60,
        )

    def test_due_five_minutes_before(self) -> None:
        settings = self.make_settings("20:00")
        now = datetime(2026, 8, 8, 19, 55, tzinfo=HONG_KONG_TIMEZONE)
        due = find_due_occurrences(settings, now, set())
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0][1].boss_id, "clama")

    def test_not_due_before_window(self) -> None:
        settings = self.make_settings("20:00")
        now = datetime(2026, 8, 8, 19, 54, tzinfo=HONG_KONG_TIMEZONE)
        self.assertEqual(find_due_occurrences(settings, now, set()), [])

    def test_due_seven_minutes_after_scheduled_time(self) -> None:
        settings = self.make_settings("22:00")
        now = datetime(2026, 8, 9, 22, 7, tzinfo=HONG_KONG_TIMEZONE)
        due = find_due_occurrences(settings, now, set())
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0][2].strftime("%H:%M"), "22:00")

    def test_not_due_at_twenty_nine_minute_deadline(self) -> None:
        settings = self.make_settings("22:00")
        now = datetime(2026, 8, 9, 22, 29, tzinfo=HONG_KONG_TIMEZONE)
        self.assertEqual(find_due_occurrences(settings, now, set()), [])

    def test_cross_midnight_window(self) -> None:
        settings = self.make_settings("00:03")
        now = datetime(2026, 8, 8, 23, 59, tzinfo=HONG_KONG_TIMEZONE)
        due = find_due_occurrences(settings, now, set())
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0][2].date().isoformat(), "2026-08-09")

    def test_completed_occurrence_is_not_due(self) -> None:
        settings = self.make_settings("20:00")
        now = datetime(2026, 8, 8, 19, 56, tzinfo=HONG_KONG_TIMEZONE)
        due = find_due_occurrences(settings, now, set())
        self.assertEqual(
            find_due_occurrences(settings, now, {due[0][0]}),
            [],
        )

    def test_boss_list_order_matches_game_ui(self) -> None:
        self.assertEqual(get_boss_definition("peri").list_index, 0)
        self.assertEqual(
            get_boss_definition("corrupted_root_beast").list_index,
            1,
        )
        self.assertEqual(get_boss_definition("clama").list_index, 2)

    def test_storage_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = BossScheduleStorage(Path(directory) / "schedule.json")
            storage.save(DEFAULT_BOSS_SCHEDULE)
            loaded = storage.load(DEFAULT_BOSS_SCHEDULE)
            self.assertEqual(loaded, DEFAULT_BOSS_SCHEDULE)


if __name__ == "__main__":
    unittest.main()

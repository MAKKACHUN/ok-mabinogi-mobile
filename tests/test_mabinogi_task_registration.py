import unittest

from src.config import config


class MabinogiTaskRegistrationTest(unittest.TestCase):
    def test_only_mabinogi_tasks_are_registered(self) -> None:
        self.assertEqual(
            config["onetime_tasks"],
            [
                [
                    "src.plugins.wild_boss.tasks.AutoWildBossTask",
                    "AutoWildBossTask",
                ],
                [
                    "src.plugins.gather.tasks.AutoGatherTask",
                    "AutoGatherTask",
                ],
            ],
        )
        self.assertEqual(config["trigger_tasks"], [])

    def test_gui_framework_features_remain_enabled(self) -> None:
        self.assertTrue(config["custom_tasks"])
        self.assertNotIn("minimal_sidebar", config)
        self.assertGreater(len(config["global_configs"]), 0)


if __name__ == "__main__":
    unittest.main()

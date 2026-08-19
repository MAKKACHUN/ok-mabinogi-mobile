import unittest

from src.gui_layout import (
    move_group_navigation_before_capture,
    move_group_task_info_below_cards,
    select_group_after_stop_notification,
    select_group_as_default_page,
)


class FakeLayout:
    def __init__(self, widgets):
        self.widgets = list(widgets)

    def removeWidget(self, widget):
        self.widgets.remove(widget)

    def addWidget(self, widget):
        self.widgets.append(widget)

    def indexOf(self, widget):
        try:
            return self.widgets.index(widget)
        except ValueError:
            return -1

    def insertWidget(self, index, widget):
        self.widgets.insert(index, widget)


class FakeTab:
    def __init__(self, group_name, widgets):
        self.group_name = group_name
        self.task_info_container = widgets[0]
        self.vBoxLayout = FakeLayout(widgets)


class FakeNavigationTab:
    def __init__(self, route_key, group_name=None):
        self.route_key = route_key
        self.group_name = group_name

    def objectName(self):
        return self.route_key


class GuiLayoutTests(unittest.TestCase):
    def test_moves_full_auto_info_below_all_task_cards(self):
        info = object()
        cards = [object(), object(), object()]
        tab = FakeTab("全自動", [info, *cards])
        main_window = type("MainWindow", (), {"grouped_task_tabs": [tab]})()

        moved = move_group_task_info_below_cards(main_window)

        self.assertTrue(moved)
        self.assertEqual(tab.vBoxLayout.widgets, [*cards, info])

    def test_leaves_other_group_pages_unchanged(self):
        info = object()
        card = object()
        tab = FakeTab("其他", [info, card])
        main_window = type("MainWindow", (), {"grouped_task_tabs": [tab]})()

        moved = move_group_task_info_below_cards(main_window)

        self.assertFalse(moved)
        self.assertEqual(tab.vBoxLayout.widgets, [info, card])

    def test_moves_full_auto_navigation_before_capture(self):
        capture_tab = FakeNavigationTab("capture")
        full_auto_tab = FakeNavigationTab("full_auto", "全自動")
        script_tab = FakeNavigationTab("script")
        item_type = type("Item", (), {})
        capture_item = item_type()
        capture_item.widget = object()
        full_auto_item = item_type()
        full_auto_item.widget = object()
        script_item = item_type()
        script_item.widget = object()
        layout = FakeLayout(
            [capture_item.widget, full_auto_item.widget, script_item.widget]
        )
        panel = type(
            "Panel",
            (),
            {
                "items": {
                    "capture": capture_item,
                    "full_auto": full_auto_item,
                    "script": script_item,
                },
                "topLayout": layout,
            },
        )()
        navigation = type("Navigation", (), {"panel": panel})()
        main_window = type(
            "MainWindow",
            (),
            {
                "start_tab": capture_tab,
                "grouped_task_tabs": [full_auto_tab],
                "navigationInterface": navigation,
            },
        )()

        moved = move_group_navigation_before_capture(main_window)

        self.assertTrue(moved)
        self.assertEqual(
            layout.widgets,
            [full_auto_item.widget, capture_item.widget, script_item.widget],
        )

    def test_moves_navigation_bar_item_without_panel_wrapper(self):
        capture_tab = FakeNavigationTab("capture")
        full_auto_tab = FakeNavigationTab("full_auto", "全自動")
        capture_widget = object()
        full_auto_widget = object()
        layout = FakeLayout([capture_widget, full_auto_widget])
        navigation = type(
            "NavigationBar",
            (),
            {
                "items": {
                    "capture": capture_widget,
                    "full_auto": full_auto_widget,
                },
                "topLayout": layout,
            },
        )()
        main_window = type(
            "MainWindow",
            (),
            {
                "start_tab": capture_tab,
                "grouped_task_tabs": [full_auto_tab],
                "navigationInterface": navigation,
            },
        )()

        moved = move_group_navigation_before_capture(main_window)

        self.assertTrue(moved)
        self.assertEqual(layout.widgets, [full_auto_widget, capture_widget])

    def test_selects_full_auto_as_default_page(self):
        capture_tab = FakeNavigationTab("capture")
        full_auto_tab = FakeNavigationTab("full_auto", "全自動")
        main_window = type(
            "MainWindow",
            (),
            {
                "grouped_task_tabs": [full_auto_tab],
                "current_tab": capture_tab,
                "switchTo": lambda self, tab: setattr(self, "current_tab", tab),
            },
        )()

        selected = select_group_as_default_page(main_window)

        self.assertTrue(selected)
        self.assertIs(main_window.current_tab, full_auto_tab)

    def test_selects_full_auto_after_stopped_notification(self):
        capture_tab = FakeNavigationTab("capture")
        full_auto_tab = FakeNavigationTab("full_auto", "全自動")
        main_window = type(
            "MainWindow",
            (),
            {
                "grouped_task_tabs": [full_auto_tab],
                "current_tab": capture_tab,
                "switchTo": lambda self, tab: setattr(self, "current_tab", tab),
            },
        )()

        selected = select_group_after_stop_notification(main_window, "Stopped")

        self.assertTrue(selected)
        self.assertIs(main_window.current_tab, full_auto_tab)

    def test_does_not_redirect_other_notifications(self):
        capture_tab = FakeNavigationTab("capture")
        full_auto_tab = FakeNavigationTab("full_auto", "全自動")
        main_window = type(
            "MainWindow",
            (),
            {
                "grouped_task_tabs": [full_auto_tab],
                "current_tab": capture_tab,
                "switchTo": lambda self, tab: setattr(self, "current_tab", tab),
            },
        )()

        selected = select_group_after_stop_notification(main_window, "Capture Error")

        self.assertFalse(selected)
        self.assertIs(main_window.current_tab, capture_tab)


if __name__ == "__main__":
    unittest.main()

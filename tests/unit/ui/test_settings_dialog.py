from aot.config.types import CameraSourceConfig, SourceType
from aot.ui.hardware_testing_panel import HardwareTestingPanel
from aot.ui.settings_dialog import SettingsDialog


def test_settings_has_named_hardware_testing_tab(qtbot) -> None:
    sources = (
        CameraSourceConfig("one", "One", SourceType.USB, True, device_index=0),
    )
    dialog = SettingsDialog(sources, lambda _sources: (), lambda: (), lambda: None)
    qtbot.addWidget(dialog)

    assert dialog.tabs.count() == 1
    assert dialog.tabs.tabText(0) == "Hardware Testing"
    assert isinstance(dialog.tabs.widget(0), HardwareTestingPanel)
    assert dialog.windowTitle() == "Settings"

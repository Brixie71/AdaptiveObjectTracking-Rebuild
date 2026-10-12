import io
import logging
import sys
import threading
import time

from aot.detection.backend import isolated_backend_output


def test_overlapping_backend_isolation_does_not_leak_or_corrupt_global_state(
    monkeypatch,
) -> None:
    original_stdout = io.StringIO()
    original_stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original_stdout)
    monkeypatch.setattr(sys, "stderr", original_stderr)
    logger = logging.getLogger("ultralytics")
    original_disabled = logger.disabled
    logger.disabled = False
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    print_second = threading.Event()

    def first():
        with isolated_backend_output():
            first_entered.set()
            release_first.wait(1.0)

    def second():
        with isolated_backend_output():
            second_entered.set()
            print_second.wait(1.0)
            print("PRIVATE-CONCURRENT-MARKER")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(0.5)
    second_thread.start()
    time.sleep(0.03)
    release_first.set()
    first_thread.join(1.0)
    assert second_entered.wait(0.5)
    print_second.set()
    second_thread.join(1.0)

    assert "PRIVATE-CONCURRENT-MARKER" not in original_stdout.getvalue()
    assert "PRIVATE-CONCURRENT-MARKER" not in original_stderr.getvalue()
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert logger.disabled is False
    logger.disabled = original_disabled

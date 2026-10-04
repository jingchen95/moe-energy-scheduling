from unittest.mock import Mock

import pytest

from src import gpu_control


def _state(clock: int) -> gpu_control.GPUClockState:
    return gpu_control.GPUClockState(0, "NVIDIA L40S", "uuid", "bus", clock)


def test_set_clock_accepts_one_l40s_clock_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_control, "supported_graphics_clocks", lambda _gpu: [2520])
    run = Mock(return_value="")
    monkeypatch.setattr(gpu_control, "_run", run)
    monkeypatch.setattr(gpu_control, "read_clock", lambda _gpu: _state(2505))

    observed = gpu_control.set_graphics_clock(0, 2520, verify_timeout_s=0)

    assert observed.graphics_clock_mhz == 2505
    run.assert_called_once_with(
        ["-i", "0", "-lgc", "2520,2520"], privileged=True
    )


def test_set_clock_rejects_more_than_one_step(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_control, "supported_graphics_clocks", lambda _gpu: [2520])
    monkeypatch.setattr(gpu_control, "_run", Mock(return_value=""))
    monkeypatch.setattr(gpu_control, "read_clock", lambda _gpu: _state(2490))
    reset = Mock(return_value=[])
    monkeypatch.setattr(gpu_control, "reset_graphics_clocks", reset)

    with pytest.raises(gpu_control.GPUControlError, match="tolerance 15 MHz"):
        gpu_control.set_graphics_clock(0, 2520, verify_timeout_s=0)

    reset.assert_called_once_with([0], verify=False)


def test_set_clock_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        gpu_control.set_graphics_clock(0, 2520, verify_tolerance_mhz=-1)


def test_set_clock_can_defer_idle_current_clock_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_control, "supported_graphics_clocks", lambda _gpu: [2520])
    run = Mock(return_value="")
    monkeypatch.setattr(gpu_control, "_run", run)
    monkeypatch.setattr(gpu_control, "read_clock", lambda _gpu: _state(2475))

    observed = gpu_control.set_graphics_clock(
        0, 2520, verify_current_clock=False
    )

    assert observed.graphics_clock_mhz == 2475
    run.assert_called_once_with(
        ["-i", "0", "-lgc", "2520,2520"], privileged=True
    )

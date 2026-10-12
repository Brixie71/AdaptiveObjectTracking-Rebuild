import cv2
import numpy as np
import pytest

from aot.capture.normalize import FrameNormalizationError, normalize_frame


def test_bgr_frame_is_preserved_when_already_canonical() -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    result = normalize_frame(frame, width=3, height=2, rotation=0)

    np.testing.assert_array_equal(result, frame)
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            np.array([[0, 255]], dtype=np.uint8),
            np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8),
        ),
        (
            np.array([[[1, 2, 3, 99], [4, 5, 6, 88]]], dtype=np.uint8),
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
        ),
    ],
)
def test_supported_input_formats_are_converted_to_bgr(frame, expected) -> None:
    result = normalize_frame(frame, width=2, height=1, rotation=0)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("rotation", "expected"),
    [
        (0, [[1, 2, 3], [4, 5, 6]]),
        (90, [[4, 1], [5, 2], [6, 3]]),
        (180, [[6, 5, 4], [3, 2, 1]]),
        (270, [[3, 6], [2, 5], [1, 4]]),
    ],
)
def test_rotation_is_applied_before_final_resize(rotation, expected) -> None:
    grayscale = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    expected_gray = np.array(expected, dtype=np.uint8)

    result = normalize_frame(
        grayscale,
        width=expected_gray.shape[1],
        height=expected_gray.shape[0],
        rotation=rotation,
    )

    np.testing.assert_array_equal(result[:, :, 0], expected_gray)


def test_result_is_resized_to_requested_canonical_shape_and_dtype() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)[:, ::2]
    assert not frame.flags.c_contiguous

    result = normalize_frame(frame, width=8, height=6, rotation=90)

    assert result.shape == (6, 8, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.array([], dtype=np.uint8),
        np.zeros((2,), dtype=np.uint8),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.zeros((1, 2, 5), dtype=np.uint8),
        np.zeros((1, 2, 3), dtype=np.float32),
    ],
)
def test_invalid_frames_are_rejected(frame) -> None:
    with pytest.raises(FrameNormalizationError):
        normalize_frame(frame, width=2, height=1, rotation=0)


@pytest.mark.parametrize("width,height", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_invalid_dimensions_are_rejected(width, height) -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    with pytest.raises(FrameNormalizationError, match="dimensions"):
        normalize_frame(frame, width=width, height=height, rotation=0)


@pytest.mark.parametrize("rotation", [-90, 45, 360])
def test_invalid_rotations_are_rejected(rotation) -> None:
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    with pytest.raises(FrameNormalizationError, match="rotation"):
        normalize_frame(frame, width=1, height=1, rotation=rotation)


def test_opencv_errors_are_translated_without_frame_contents(monkeypatch) -> None:
    frame = np.full((1, 1, 3), 173, dtype=np.uint8)

    def fail_resize(*_args, **_kwargs):
        raise cv2.error("secret frame content 173")

    monkeypatch.setattr(cv2, "resize", fail_resize)

    with pytest.raises(FrameNormalizationError) as caught:
        normalize_frame(frame, width=2, height=2, rotation=0)

    assert "173" not in str(caught.value)

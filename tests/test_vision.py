"""Unit tests for Vision Pipeline (src/vision)."""

import numpy as np
import pytest
from src.vision.detector import HandLandmarkDetector, LandmarkFrame
from src.vision.visualizer import HandVisualizer


def test_landmark_normalization_wrist_centered():
    """Verify wrist anchor (index 0) is translated to origin (0, 0, 0)."""
    raw_coords = np.zeros((21, 3), dtype=np.float32)
    raw_coords[0] = [10.0, 20.0, 30.0]  # Wrist offset
    raw_coords[9] = [10.0, 25.0, 30.0]  # Middle MCP offset (dist = 5.0)

    normalized = HandLandmarkDetector.normalize_coordinates(raw_coords)

    # Wrist must be at (0, 0, 0)
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)
    # Distance between wrist and Middle MCP must be exactly 1.0 (unit normalized)
    palm_len = np.linalg.norm(normalized[9])
    assert abs(palm_len - 1.0) < 1e-5


def test_hand_visualizer():
    """Verify HandVisualizer draws on frames without error."""
    visualizer = HandVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock landmark points normalized [0, 1]
    mock_lms = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    annotated = visualizer.draw_skeleton(frame, mock_lms)

    assert annotated.shape == frame.shape
    # Ensure drawing modified the frame pixels
    assert np.any(annotated > 0)


def test_detector_blank_frame_no_detection():
    """Verify that a black frame gracefully produces detected=False without crashing."""
    detector = HandLandmarkDetector()
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = detector.process_frame(blank_frame)

    assert isinstance(result, LandmarkFrame)
    assert not result.detected
    assert result.raw_landmarks is None
    detector.close()

"""Hand Landmark Extraction using MediaPipe Tasks Vision.

Authored for the Vision Pipeline (Lead: Debonnita).
Provides high-performance, robust 3D hand tracking (21 keypoints)
with relative coordinate normalization (wrist anchor and palm scale).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Landmark index definitions for reference
HAND_LANDMARK_NAMES = [
    "WRIST",  # 0
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",  # 1-4
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",  # 5-8
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",  # 9-12
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",  # 13-16
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"  # 17-20
]

# Standard hand joint linkages for visualization & connectivity
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17)     # Palm base
]


@dataclass
class LandmarkFrame:
    """Represents a single frame's detected hand landmarks."""
    frame_idx: int
    timestamp_sec: float
    detected: bool
    # Raw coordinates (21, 3): [x, y, z] normalized to image coordinates
    raw_landmarks: Optional[np.ndarray] = None
    # World coordinates (21, 3) in meters, origin at hand center
    world_landmarks: Optional[np.ndarray] = None
    # Scale- and translation-invariant normalized coordinates (21, 3)
    normalized_landmarks: Optional[np.ndarray] = None
    handedness: str = "Unknown"  # "Left" or "Right"
    confidence: float = 0.0


class HandLandmarkDetector:
    """Detects 21 3D hand landmarks in video frames using MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe HandLandmarker.

        Args:
            model_path: Path to hand_landmarker.task. Defaults to models/hand_landmarker.task.
            num_hands: Maximum number of hands to detect.
            min_hand_detection_confidence: Confidence threshold for hand detection.
            min_hand_presence_confidence: Confidence threshold for hand presence.
            min_tracking_confidence: Confidence threshold for tracking.
        """
        if model_path is None:
            # Look in standard repository locations
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_path = repo_root / "models" / "hand_landmarker.task"
            if default_path.exists():
                model_path = str(default_path)
            else:
                raise FileNotFoundError(
                    f"MediaPipe task model not found at {default_path}. "
                    "Ensure models/hand_landmarker.task is present."
                )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_counter = 0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: Optional[float] = None
    ) -> LandmarkFrame:
        """Process an OpenCV BGR image and extract 3D landmarks.

        Args:
            frame_bgr: OpenCV image array (H, W, 3).
            timestamp_sec: Optional timestamp in seconds.

        Returns:
            LandmarkFrame dataclass with raw, world, and normalized coordinates.
        """
        if timestamp_sec is None:
            timestamp_sec = self.frame_counter / 30.0

        # Convert OpenCV BGR to MediaPipe Image (RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks:
            frame_res = LandmarkFrame(
                frame_idx=self.frame_counter,
                timestamp_sec=timestamp_sec,
                detected=False,
            )
            self.frame_counter += 1
            return frame_res

        # Take primary hand (index 0)
        primary_landmarks = result.hand_landmarks[0]
        raw_coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in primary_landmarks],
            dtype=np.float32
        )

        world_coords = None
        if result.hand_world_landmarks:
            primary_world = result.hand_world_landmarks[0]
            world_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in primary_world],
                dtype=np.float32
            )

        # Compute translation- and scale-normalized coordinates
        normalized_coords = self.normalize_coordinates(raw_coords)

        handedness = "Unknown"
        confidence = 0.0
        if result.handedness:
            category = result.handedness[0][0]
            handedness = category.category_name
            confidence = float(category.score)

        frame_result = LandmarkFrame(
            frame_idx=self.frame_counter,
            timestamp_sec=timestamp_sec,
            detected=True,
            raw_landmarks=raw_coords,
            world_landmarks=world_coords if world_coords is not None else raw_coords,
            normalized_landmarks=normalized_coords,
            handedness=handedness,
            confidence=confidence,
        )
        self.frame_counter += 1
        return frame_result

    @staticmethod
    def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates for scale and translation invariance.

        1. Translation invariance: Shifts origin (0, 0, 0) to Wrist (index 0).
        2. Scale invariance: Scales coordinates by the distance between
           Wrist (0) and Middle Finger MCP (9) (palm length).

        Args:
            coords: Array of shape (21, 3).

        Returns:
            Normalized coordinates array of shape (21, 3).
        """
        wrist = coords[0]
        centered = coords - wrist

        palm_length = np.linalg.norm(centered[9])  # Middle MCP is index 9
        if palm_length < 1e-6:
            scale_factor = 1.0
        else:
            scale_factor = palm_length

        return centered / scale_factor

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

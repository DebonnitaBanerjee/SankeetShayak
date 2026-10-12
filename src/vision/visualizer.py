"""Vision utility: Landmark and Skeletal Visualizer.

Renders 21 3D hand joints and connecting bone linkages onto OpenCV frames.
"""

from typing import Optional
import numpy as np
import cv2
from src.vision.detector import HAND_CONNECTIONS


class HandVisualizer:
    """Draws hand skeletons and kinematic overlays onto video frames."""

    def __init__(self):
        self.connections = HAND_CONNECTIONS

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        raw_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw bone linkages and joint landmarks on a frame.

        Args:
            frame_bgr: BGR frame array (H, W, 3).
            raw_landmarks: Optional (21, 3) normalized [0, 1] landmark coordinates.

        Returns:
            Annotated frame array.
        """
        output = frame_bgr.copy()
        if raw_landmarks is None or len(raw_landmarks) != 21:
            return output

        h, w = output.shape[:2]

        # Draw bone connections
        for start_idx, end_idx in self.connections:
            p1 = (int(raw_landmarks[start_idx, 0] * w), int(raw_landmarks[start_idx, 1] * h))
            p2 = (int(raw_landmarks[end_idx, 0] * w), int(raw_landmarks[end_idx, 1] * h))
            cv2.line(output, p1, p2, (0, 255, 128), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx in range(21):
            px = int(raw_landmarks[idx, 0] * w)
            py = int(raw_landmarks[idx, 1] * h)
            # Differentiate fingertips with orange circle
            color = (0, 140, 255) if idx in (4, 8, 12, 16, 20) else (255, 200, 0)
            radius = 5 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(output, (px, py), radius, color, -1, cv2.LINE_AA)

        return output

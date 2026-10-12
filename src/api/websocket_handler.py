"""WebSocket Streaming Handler with Real-Time Sliding Buffer.

Authored for the API & Backend Workstream (Lead: Harsimran).
Maintains per-client circular frame buffers to continuously evaluate
kinematics, dynamics, and sign predictions over live WebSocket connections.
"""

from collections import deque
import json
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from src.models.predictor import GesturePredictor


class StreamSession:
    """Manages an active real-time gesture streaming session."""

    def __init__(self, buffer_size: int = 30, fps: float = 30.0, predictor: Optional[GesturePredictor] = None):
        self.buffer_size = buffer_size
        self.fps = fps
        self.buffer: deque = deque(maxlen=buffer_size)
        self.predictor = predictor or GesturePredictor(fps=fps)
        self.frame_count = 0

    def push_frame(self, landmarks_21x3: np.ndarray) -> Optional[Dict]:
        """Add a frame to the buffer and evaluate if buffer is sufficiently full.

        Args:
            landmarks_21x3: Array of shape (21, 3).

        Returns:
            Dictionary with telemetry and prediction, or None if buffer warming up.
        """
        self.buffer.append(landmarks_21x3)
        self.frame_count += 1

        # We need at least 15 frames to compute meaningful derivatives
        if len(self.buffer) < 15:
            return None

        traj_window = np.array(self.buffer, dtype=np.float64)
        prediction = self.predictor.predict_trajectory(traj_window)

        return {
            "type": "inference_update",
            "frame_idx": self.frame_count,
            "prediction": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prediction.probabilities.items()},
            "telemetry": {k: round(v, 4) for k, v in prediction.telemetry.items()},
        }


class WebSocketStreamManager:
    """Coordinates active WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._shared_predictor: Optional[GesturePredictor] = None

    @property
    def predictor(self) -> GesturePredictor:
        if self._shared_predictor is None:
            self._shared_predictor = GesturePredictor()
        return self._shared_predictor

    async def connect(self, websocket: WebSocket) -> StreamSession:
        await websocket.accept()
        self.active_connections.append(websocket)
        return StreamSession(predictor=self.predictor)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

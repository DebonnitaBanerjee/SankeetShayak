"""Pydantic data schemas for REST API and WebSocket streams."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    project: str = "SankeetShayak"


class SignClassListResponse(BaseModel):
    classes: List[str]
    count: int


class TrajectoryPredictRequest(BaseModel):
    # Shape: (T, 21, 3) normalized hand landmark coordinates
    trajectory: List[List[List[float]]] = Field(
        ...,
        description="Time series of 3D hand landmark coordinates (T, 21, 3)"
    )
    fps: float = Field(default=30.0, ge=1.0, le=120.0)


class TelemetryData(BaseModel):
    peak_speed_m_s: float
    mean_speed_m_s: float
    peak_accel_m_s2: float
    peak_jerk_m_s3: float
    peak_ke_joules: float
    total_energy_j_s: float
    peak_force_n: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    telemetry: TelemetryData


class StreamFramePayload(BaseModel):
    frame_idx: int
    timestamp_sec: Optional[float] = None
    # 21 landmarks [x, y, z]
    landmarks: List[List[float]]

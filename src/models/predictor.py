"""Sign Gesture Inference Engine.

Authored for the ML & Backend Workstream (Lead: Harsimran).
Provides fast, production-ready inference given sliding-window hand trajectories.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np

from src.data.aggregator import FeatureAggregator


@dataclass
class GesturePrediction:
    """Prediction result and accompanying physical telemetry."""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    telemetry: Dict[str, float]


class GesturePredictor:
    """Predicts sign labels from sliding window coordinate trajectories."""

    def __init__(self, model_path: Optional[str] = None, fps: float = 30.0):
        self.fps = fps
        self.aggregator = FeatureAggregator(fps=fps)

        if model_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_path = repo_root / "models" / "handshayak_model.joblib"
            model_path = str(default_path)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            # Automatically build model if not already trained
            from src.models.train import build_and_train_model
            self.model = build_and_train_model(output_model_path=str(self.model_path))
        else:
            self.model = joblib.load(self.model_path)

    def predict_trajectory(self, trajectory: np.ndarray) -> GesturePrediction:
        """Run inference on a gesture trajectory.

        Args:
            trajectory: Array of shape (T, 21, 3) normalized hand landmarks.

        Returns:
            GesturePrediction containing predicted sign and telemetry.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[1] != 21:
            raise ValueError(f"Trajectory must have shape (T, 21, 3). Got {traj.shape}")

        # Extract combined features
        features = self.aggregator.extract_features(traj, fps=self.fps)
        comb_vec = features["combined"].reshape(1, -1)

        # Predict
        predicted_label = str(self.model.predict(comb_vec)[0])

        # Probabilities
        probs = {}
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba_arr = self.model.predict_proba(comb_vec)[0]
            classes = self.model.classes_
            probs = {str(c): float(p) for c, p in zip(classes, proba_arr)}
            confidence = float(np.max(proba_arr))

        # Extract telemetry summary from dynamics
        dyn_vec = features["dynamics"]
        telemetry = {
            "peak_speed_m_s": float(dyn_vec[0]),
            "mean_speed_m_s": float(dyn_vec[1]),
            "peak_accel_m_s2": float(dyn_vec[7]),
            "peak_jerk_m_s3": float(dyn_vec[9]),
            "peak_ke_joules": float(dyn_vec[11]),
            "total_energy_j_s": float(dyn_vec[13]),
            "peak_force_n": float(dyn_vec[14]),
        }

        return GesturePrediction(
            label=predicted_label,
            confidence=confidence,
            probabilities=probs,
            telemetry=telemetry,
        )

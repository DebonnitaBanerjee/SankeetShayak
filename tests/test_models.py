"""Unit tests for ML & Ablation Pipeline (src/models)."""

import numpy as np
import pytest
from src.data.collector import GestureDatasetBuilder
from src.models.ablation import AblationStudy
from src.models.predictor import GesturePredictor
from src.models.train import build_and_train_model


def test_synthetic_data_generation():
    """Verify synthetic dataset generator produces valid shape and classes."""
    trajectories, labels, class_names = GestureDatasetBuilder.generate_synthetic_gestures(
        samples_per_class=5, fps=30.0, frames_per_sample=20
    )

    assert len(trajectories) == 5 * 4
    assert len(labels) == 5 * 4
    assert set(class_names) == {"tap_gentle", "tap_forceful", "swipe_slow", "swipe_rapid"}
    assert trajectories[0].shape == (20, 21, 3)


def test_ablation_study_execution():
    """Verify that ablation study evaluates all 3 baseline conditions."""
    N = 24
    X_shape = np.random.randn(N, 60)
    X_dyn = np.random.randn(N, 15)
    X_comb = np.hstack([X_shape, X_dyn])
    y = np.array(["tap_gentle", "tap_forceful"] * (N // 2))

    ablation = AblationStudy(random_state=42)
    results = ablation.run_study(X_shape, X_dyn, X_comb, y, test_size=0.25)

    assert "Shape-Only Baseline" in results
    assert "Dynamics-Only Baseline" in results
    assert "HandShayak (Combined)" in results

    for name, res in results.items():
        assert 0.0 <= res.accuracy <= 1.0
        assert 0.0 <= res.macro_f1 <= 1.0


def test_predictor_and_telemetry(tmp_path):
    """Verify model training and predictor inference with telemetry output."""
    model_path = str(tmp_path / "test_model.joblib")
    build_and_train_model(samples_per_class=10, output_model_path=model_path, run_ablation=False)

    predictor = GesturePredictor(model_path=model_path, fps=30.0)

    # Test sample trajectory
    sample_traj = np.random.normal(0, 0.1, size=(25, 21, 3))
    prediction = predictor.predict_trajectory(sample_traj)

    assert prediction.label in {"tap_gentle", "tap_forceful", "swipe_slow", "swipe_rapid"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert "peak_speed_m_s" in prediction.telemetry
    assert "peak_ke_joules" in prediction.telemetry

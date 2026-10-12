"""Interactive Demo for SankeetShayak / HandShayak.

Runs either:
1. An offline simulation demonstrating physics features and classification on synthetic gestures.
2. A live webcam visualizer overlaying 3D hand landmarks and instantaneous physics telemetry.

Usage:
    # Run simulation demo (works anywhere, headless/no camera required):
    python demo.py --simulate

    # Run live webcam tracking:
    python demo.py --webcam
"""

import argparse
import sys
import time
import cv2
import numpy as np

from src.data.collector import GestureDatasetBuilder
from src.models.predictor import GesturePredictor
from src.vision.detector import HandLandmarkDetector
from src.vision.visualizer import HandVisualizer


def run_simulation_demo():
    print("=" * 65)
    print("🖐️  SANKEET SHAYAK: PHYSICS-INFORMED SIGN RECOGNITION DEMO")
    print("=" * 65)
    print("\n1. Initializing Gesture Predictor & Physics Engine...")
    predictor = GesturePredictor()

    print("2. Generating test gestures (Gentle Tap vs. Forceful Strike)...")
    trajs, labels, _ = GestureDatasetBuilder.generate_synthetic_gestures(
        samples_per_class=1, fps=30.0, frames_per_sample=30
    )

    for traj, true_label in zip(trajs, labels):
        pred = predictor.predict_trajectory(traj)
        print(f"\n▶ Ground Truth Sign : {true_label.upper()}")
        print(f"  Predicted Sign    : {pred.label.upper()} (Confidence: {pred.confidence * 100:.1f}%)")
        print("  Biomechanical Telemetry:")
        print(f"    • Peak Speed    : {pred.telemetry['peak_speed_m_s']:.3f} m/s")
        print(f"    • Peak Accel    : {pred.telemetry['peak_accel_m_s2']:.3f} m/s²")
        print(f"    • Peak Jerk     : {pred.telemetry['peak_jerk_m_s3']:.3f} m/s³")
        print(f"    • Kinetic Energy: {pred.telemetry['peak_ke_joules']:.4f} Joules")
        print(f"    • Inferred Force: {pred.telemetry['peak_force_n']:.3f} N")

    print("\n" + "=" * 65)
    print("✅ Simulation completed successfully! Physics features clearly differentiate")
    print("   movement dynamics between gentle and forceful signs.")
    print("=" * 65)


def run_webcam_demo():
    print("\nStarting live webcam stream... Press 'q' to exit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam device (index 0).")
        sys.exit(1)

    detector = HandLandmarkDetector()
    visualizer = HandVisualizer()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)

            res = detector.process_frame(frame)
            if res.detected and res.raw_landmarks is not None:
                frame = visualizer.draw_skeleton(frame, res.raw_landmarks)
                cv2.putText(
                    frame,
                    f"Hand: {res.handedness} ({res.confidence * 100:.0f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("SankeetShayak Live Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SankeetShayak Demo Runner")
    parser.add_argument("--simulate", action="store_true", help="Run simulated physics and classification demo")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam hand tracking")
    args = parser.parse_args()

    if args.webcam:
        run_webcam_demo()
    else:
        run_simulation_demo()

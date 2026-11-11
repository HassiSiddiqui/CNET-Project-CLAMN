"""CL-ANM Fault Detector.

Predicts fault posterior p_t over:
  ['none', 'link', 'ddos', 'misconfig', 'cascade']
using Gradient Boosted Trees (HistGradientBoosting / XGBoost) or MLP over rolling telemetry features.
Triggers autonomous investigation / remediation when max posterior >= threshold (default 0.75).
"""

from __future__ import annotations
import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


FAULT_CLASSES = ["none", "link", "ddos", "misconfig", "cascade"]
CLASS_TO_IDX = {c: i for i, c in enumerate(FAULT_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(FAULT_CLASSES)}


@dataclass
class DetectionResult:
    predicted_class: str
    posterior: Dict[str, float]
    confidence: float
    is_anomaly: bool
    features: List[float]


class FaultDetector:
    """Predicts real-time posterior probabilities over network fault states."""

    def __init__(
        self,
        model_path: Optional[str] = "results/checkpoints/detector.pkl",
        threshold: float = 0.75,
        random_state: int = 42,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.random_state = random_state
        self.model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            random_state=random_state,
        )
        self.is_trained: bool = False
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @staticmethod
    def extract_features_from_window(records: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregate a 5-step rolling window of telemetry records into a feature vector."""
        if not records:
            return np.zeros(25, dtype=np.float32)

        metrics = [
            [
                r.get("rx_pps", 0.0),
                r.get("tx_pps", 0.0),
                r.get("loss_rate", 0.0),
                r.get("utilization", 0.0),
                r.get("latency_ms", 1.0),
                r.get("syn_ack_ratio", 1.0),
                r.get("src_entropy", 0.0),
                r.get("unique_dsts", 1.0),
                r.get("cpu_percent", 0.0),
                r.get("mem_percent", 0.0),
            ]
            for r in records
        ]
        arr = np.array(metrics, dtype=np.float32)  # shape (W, 10)

        # Means and Maxima
        means = np.mean(arr, axis=0)
        maxs = np.max(arr, axis=0)
        # Rates of change (diff between last and first in window)
        diffs = arr[-1] - arr[0] if len(arr) > 1 else np.zeros_like(means)

        feat = np.concatenate([means, maxs[:4], diffs[:5], [float(len(records))]])
        return feat.astype(np.float32)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fit detector on labeled feature matrix."""
        self.model.fit(X, y)
        self.is_trained = True

        train_preds = self.model.predict(X)
        train_acc = float(accuracy_score(y, train_preds))
        train_f1 = float(f1_score(y, train_preds, average="weighted", zero_division=0))

        val_acc, val_f1 = 0.0, 0.0
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = float(accuracy_score(y_val, val_preds))
            val_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

        if self.model_path:
            self.save(self.model_path)

        return {
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }

    def predict(self, feature_vector: np.ndarray) -> DetectionResult:
        """Infers fault posterior distribution for a single telemetry window."""
        if not self.is_trained:
            # Fallback heuristic rule detector if model not yet trained
            return self._heuristic_fallback(feature_vector)

        feat = feature_vector.reshape(1, -1)
        probs = self.model.predict_proba(feat)[0]
        # Align with FAULT_CLASSES
        posterior: Dict[str, float] = {}
        classes = self.model.classes_
        for idx, cls_id in enumerate(classes):
            cls_name = IDX_TO_CLASS.get(int(cls_id), "none")
            posterior[cls_name] = float(probs[idx])
        for c in FAULT_CLASSES:
            posterior.setdefault(c, 0.0)

        best_cls = max(posterior, key=posterior.get)
        confidence = posterior[best_cls]
        is_anomaly = (best_cls != "none") and (confidence >= self.threshold)

        return DetectionResult(
            predicted_class=best_cls,
            posterior=posterior,
            confidence=confidence,
            is_anomaly=is_anomaly,
            features=feature_vector.tolist(),
        )

    def _heuristic_fallback(self, feat: np.ndarray) -> DetectionResult:
        """Deterministic rule-based baseline posterior when weights are uninitialized."""
        rx_pps_mean = feat[0] if len(feat) > 0 else 0.0
        loss_rate_mean = feat[2] if len(feat) > 2 else 0.0
        latency_mean = feat[4] if len(feat) > 4 else 1.0
        syn_ack_ratio = feat[5] if len(feat) > 5 else 1.0

        p = {"none": 0.8, "link": 0.05, "ddos": 0.05, "misconfig": 0.05, "cascade": 0.05}
        if loss_rate_mean > 0.5:
            p = {"none": 0.05, "link": 0.80, "ddos": 0.05, "misconfig": 0.05, "cascade": 0.05}
        elif syn_ack_ratio > 10.0 or rx_pps_mean > 10000.0:
            p = {"none": 0.05, "link": 0.05, "ddos": 0.85, "misconfig": 0.02, "cascade": 0.03}
        elif latency_mean > 100.0:
            p = {"none": 0.10, "link": 0.10, "ddos": 0.05, "misconfig": 0.70, "cascade": 0.05}

        best_cls = max(p, key=p.get)
        confidence = p[best_cls]
        return DetectionResult(
            predicted_class=best_cls,
            posterior=p,
            confidence=confidence,
            is_anomaly=(best_cls != "none" and confidence >= self.threshold),
            features=feat.tolist(),
        )

    def save(self, path: str) -> None:
        """Persist detector model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str) -> None:
        """Load persisted detector model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


def generate_synthetic_detector_data(n_samples: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic telemetry feature vectors matching S1-S5 fault dynamics."""
    rng = np.random.default_rng(seed)
    X = []
    y = []
    samples_per_cls = n_samples // len(FAULT_CLASSES)

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        for _ in range(samples_per_cls):
            feat = np.zeros(20, dtype=np.float32)
            noise_scale = rng.uniform(0.05, 0.25)
            if cls_name == "none":
                feat[0] = rng.uniform(100, 800)  # rx_pps with burstiness
                feat[2] = rng.uniform(0.0, 0.03)  # occasional transient loss
                feat[3] = rng.uniform(0.1, 0.6)  # util
                feat[4] = rng.uniform(1.0, 15.0)  # latency
                feat[5] = rng.uniform(0.7, 1.4)  # syn/ack
                feat[6] = rng.uniform(2.0, 3.8)  # entropy
            elif cls_name == "link":
                feat[0] = rng.uniform(0, 150)  # degraded/collapsed pps
                feat[2] = rng.uniform(0.6, 1.0)  # realistic flapping/hard loss
                feat[3] = rng.uniform(0.0, 0.3)
                feat[4] = rng.uniform(30.0, 180.0)  # high latency
                feat[5] = rng.uniform(0.8, 1.5)
                feat[6] = rng.uniform(1.0, 2.5)
            elif cls_name == "ddos":
                feat[0] = rng.uniform(4000, 25000)  # varying intensity flood pps
                feat[2] = rng.uniform(0.1, 0.65)  # buffer overflow loss
                feat[3] = rng.uniform(0.85, 1.0)  # high util
                feat[4] = rng.uniform(15.0, 90.0)
                feat[5] = rng.uniform(5.0, 100.0)  # high syn/ack ratio
                feat[6] = rng.uniform(0.1, 1.2)  # concentrated src-entropy
            elif cls_name == "misconfig":
                feat[0] = rng.uniform(30, 400)
                feat[2] = rng.uniform(0.3, 0.85)  # dropped by ACL / blackhole / MTU
                feat[3] = rng.uniform(0.1, 0.5)
                feat[4] = rng.uniform(20.0, 140.0)
                feat[5] = rng.uniform(0.8, 1.8)
                feat[6] = rng.uniform(1.5, 3.0)
            elif cls_name == "cascade":
                feat[0] = rng.uniform(400, 2500)
                feat[2] = rng.uniform(0.5, 0.95)  # escalating loss
                feat[3] = rng.uniform(0.8, 1.0)
                feat[4] = rng.uniform(60.0, 320.0)
                feat[5] = rng.uniform(1.0, 3.0)
                feat[6] = rng.uniform(1.2, 2.8)

            # Realistic Gaussian noise across all telemetry channels
            feat += rng.normal(0.0, noise_scale, size=20)
            feat = np.clip(feat, 0.0, None)
            X.append(feat)
            y.append(cls_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="CL-ANM Fault Detector CLI")
    parser.add_argument("--train", action="store_true", help="Train on synthetic/historical dataset")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model-path", type=str, default="results/checkpoints/detector.pkl")
    args = parser.parse_args()

    detector = FaultDetector(model_path=args.model_path)
    if args.train:
        X, y = generate_synthetic_detector_data(args.samples)
        # Split 80/20 train/val
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        res = detector.train(X_train, y_train, X_val, y_val)
        print(f"Trained Detector: Val Acc = {res['val_acc']*100:.2f}%, Val F1 = {res['val_f1']:.3f}")
    else:
        sample_feat = np.array([12000.0, 100.0, 0.35, 1.0, 45.0, 35.0, 0.2] + [0.0]*13, dtype=np.float32)
        out = detector.predict(sample_feat)
        print(f"Sample Prediction: {out.predicted_class} (Confidence: {out.confidence:.3f})")


if __name__ == "__main__":
    main()


"""Smoke tests for FaultDetector and LogParser."""

import os
import numpy as np
import pytest
from src.detector import FaultDetector, generate_synthetic_detector_data, FAULT_CLASSES
from src.log_parser import LogParser, ParsedLogEvent


def test_log_parser_known_patterns():
    parser = LogParser(use_embeddings=False)
    ev1 = parser.parse_line("ovs|0001|bridge|INFO|link s1-eth1 changed state to DOWN")
    assert ev1.event_type == "link_down"
    assert ev1.interface == "s1-eth1"
    assert ev1.device == "s1"
    assert ev1.severity == "ERROR"

    ev2 = parser.parse_line("drop packets on s2-eth2 due to congestion")
    assert ev2.event_type == "port_drop"


def test_log_parser_drain3_unstructured():
    parser = LogParser(use_embeddings=False)
    ev = parser.parse_line("BGP peer 192.168.1.1 reached maximum flap threshold 50")
    assert ev.event_type == "unstructured"
    assert ev.cluster_id >= 0
    assert len(ev.template) > 0


def test_detector_heuristic_fallback():
    detector = FaultDetector(model_path=None)
    # High loss feature vector
    feat_high_loss = np.array([10.0, 5.0, 0.95, 0.2, 120.0, 1.0] + [0.0]*14, dtype=np.float32)
    res = detector.predict(feat_high_loss)
    assert res.predicted_class == "link"
    assert res.confidence >= 0.75
    assert res.is_anomaly is True


def test_detector_train_and_predict(tmp_path):
    model_file = str(tmp_path / "detector_test.pkl")
    detector = FaultDetector(model_path=model_file, threshold=0.70)
    X, y = generate_synthetic_detector_data(n_samples=250, seed=42)

    res = detector.train(X, y, X_val=X[:50], y_val=y[:50])
    assert res["train_acc"] > 0.85
    assert detector.is_trained
    assert os.path.exists(model_file)

    # Predict on a DDoS-like vector
    ddos_vec = np.zeros(20, dtype=np.float32)
    ddos_vec[0] = 15000.0  # high pps
    ddos_vec[2] = 0.4  # loss
    ddos_vec[3] = 1.0  # max util
    ddos_vec[5] = 50.0  # high syn/ack ratio
    ddos_vec[6] = 0.2  # low entropy

    pred = detector.predict(ddos_vec)
    assert pred.predicted_class in FAULT_CLASSES
    assert sum(pred.posterior.values()) == pytest.approx(1.0, abs=1e-2)


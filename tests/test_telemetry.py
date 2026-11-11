"""Smoke tests for TelemetryCollector."""

import numpy as np
import pytest
from src.telemetry_collector import TelemetryCollector, TelemetryRecord


def test_telemetry_collector_dry_run():
    collector = TelemetryCollector(switches=["s1", "s2"], dry_run=True)
    records = collector.collect_step(step=0)
    assert len(records) > 0
    rec = records[0]
    assert rec.switch_id in ["s1", "s2"]
    assert rec.staleness_mask == 0
    assert rec.cpu_percent >= 0.0


def test_counter_reset_handling():
    collector = TelemetryCollector(dry_run=True)
    # First reading: time=100.0, pkts=1000
    r1, _, _, _ = collector.compute_rate("s1", 1, 100.0, 1000, 1000, 10000, 10000)
    assert r1 == 0.0

    # Second reading: normal increment
    r2, _, _, _ = collector.compute_rate("s1", 1, 101.0, 1100, 1100, 11000, 11000)
    assert abs(r2 - 100.0) < 1e-2

    # Third reading: counter reset (dropped to 50 pkts)
    r3, _, _, _ = collector.compute_rate("s1", 1, 102.0, 50, 50, 500, 500)
    assert r3 == 50.0  # Does not produce negative rates!


def test_entropy_computation():
    # Uniform distribution of 4 IPs
    ips_uniform = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    ent_uniform = TelemetryCollector.compute_entropy(ips_uniform)
    assert abs(ent_uniform - 2.0) < 1e-3

    # Concentrated single IP (DDoS signature)
    ips_ddos = ["10.0.0.1"] * 100
    ent_ddos = TelemetryCollector.compute_entropy(ips_ddos)
    assert ent_ddos == 0.0


def test_robust_scaler():
    collector = TelemetryCollector(dry_run=True)
    data = np.array([
        [10.0, 100.0],
        [12.0, 110.0],
        [11.0, 105.0],
        [13.0, 115.0],
        [100.0, 1000.0],  # Outlier
    ])
    feature_names = ["feat1", "feat2"]
    collector.fit_scaler(data, feature_names)
    assert collector.fitted
    assert "feat1" in collector.scaler_stats

    transformed = collector.transform(data, feature_names)
    assert transformed.shape == data.shape
    # Outliers should not be clamped / clipped
    assert transformed[4, 0] > 5.0


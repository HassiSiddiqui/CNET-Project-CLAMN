"""conftest.py — Pytest config: marks and fixtures."""
import pytest

def pytest_configure(config):
    config.addinivalue_line("markers", "needs_root: test requires root/Mininet access")

def pytest_collection_modifyitems(config, items):
    """Skip needs_root tests unless --run-root flag passed."""
    if config.getoption("--run-root", default=False):
        return
    skip_root = pytest.mark.skip(reason="needs root/Mininet; pass --run-root to enable")
    for item in items:
        if "needs_root" in item.keywords:
            item.add_marker(skip_root)

def pytest_addoption(parser):
    parser.addoption("--run-root", action="store_true", default=False,
                     help="Run tests that require root/Mininet access")


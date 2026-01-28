"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def mock_settings(monkeypatch):
    """Provide mock settings for testing."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TASE_API_KEY", "test-tase-key")
    
    from src.config import Settings
    return Settings()


@pytest.fixture
def sample_symbols():
    """Provide sample TASE stock symbols."""
    return [
        "TEVA.TA",
        "NICE.TA",
        "CHKP.TA",
        "ICL.TA",
        "POLI.TA",
    ]

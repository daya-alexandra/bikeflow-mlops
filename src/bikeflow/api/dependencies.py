"""FastAPI dependencies for replaceable runtime components."""

from functools import lru_cache

from bikeflow.config import get_settings
from bikeflow.model.protocol import Predictor
from bikeflow.model.stub import StubPredictor


@lru_cache
def get_predictor() -> Predictor:
    """Provide the current predictor implementation."""

    return StubPredictor(get_settings().stub_prediction)

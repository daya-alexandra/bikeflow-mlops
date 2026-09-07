"""Centralized preliminary request and response schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from bikeflow.model.protocol import FeatureValue


class Season(StrEnum):
    """Meteorological season derived from prediction time."""

    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


def season_from_month(month: int) -> Season:
    """Map a calendar month to a meteorological season."""

    if month in (12, 1, 2):
        return Season.WINTER
    if month in (3, 4, 5):
        return Season.SPRING
    if month in (6, 7, 8):
        return Season.SUMMER
    return Season.AUTUMN


class PredictionRequest(BaseModel):
    """Preliminary Seoul Bike Sharing feature contract for one prediction."""

    model_config = ConfigDict(extra="forbid")

    prediction_time: datetime
    temperature_c: float = Field(ge=-100, le=80)
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_m_s: float = Field(ge=0)
    visibility_10m: int = Field(ge=0)
    dew_point_c: float = Field(ge=-100, le=80)
    solar_radiation_mj_m2: float = Field(ge=0)
    rainfall_mm: float = Field(ge=0)
    snowfall_cm: float = Field(ge=0)
    holiday: bool
    functioning_day: bool

    def to_features(self) -> dict[str, FeatureValue]:
        """Build one model row, deriving hour and season from prediction_time."""

        features: dict[str, FeatureValue] = self.model_dump(exclude={"prediction_time"})
        features["hour"] = self.prediction_time.hour
        features["season"] = season_from_month(self.prediction_time.month).value
        return features


class PredictionResponse(BaseModel):
    """Response returned by the demand endpoint."""

    prediction_time: datetime
    predicted_rentals: float = Field(ge=0)
    model_version: str


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str

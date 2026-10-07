from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.domain.models import PlanStyle
from travel_agent.domain.tool_models import RouteMode


class OptimizationSolveStatus(StrEnum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    DEGRADED = "degraded"
    INFEASIBLE = "infeasible"


class OptimizationBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_solve_ms: int = Field(default=800, ge=10, le=30_000)
    max_search_states: int = Field(default=20_000, ge=1, le=1_000_000)
    candidate_limit: int = Field(default=8, ge=1, le=20)
    variant_count: int = Field(default=3, ge=1, le=3)


class ObjectiveWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    preference: int = Field(ge=0)
    diversity: int = Field(ge=0)
    travel: int = Field(ge=0)
    cost: int = Field(ge=0)
    proximity: int = Field(default=4, ge=0)


class OptimizationPOI(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    categories: tuple[str, ...]
    duration_minutes: int = Field(gt=0)
    party_cost: Decimal | None = Field(default=None, ge=0)
    preference_value: int
    data_confidence: float = Field(ge=0, le=1)
    must_visit: bool
    available_days: tuple[date, ...]


class RouteMatrixEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    origin_id: str
    destination_id: str
    duration_minutes: int = Field(gt=0)
    distance_meters: int = Field(gt=0)
    mode: RouteMode
    provider: str
    data_confidence: float = Field(ge=0, le=1)


class OptimizationProblem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    dates: tuple[date, ...]
    anchor_id: str
    pois: tuple[OptimizationPOI, ...]
    route_matrix: tuple[RouteMatrixEntry, ...]
    total_budget: Decimal | None = Field(default=None, gt=0)
    max_daily_activity_minutes: int = Field(gt=0)
    max_daily_walking_meters: int = Field(ge=0)
    max_walking_leg_meters: int = Field(gt=0)
    available_minutes_by_day: dict[date, int]
    weights_by_style: dict[PlanStyle, ObjectiveWeights]
    budget: OptimizationBudget


class OptimizationDayAssignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    poi_ids: tuple[str, ...]


class ObjectiveBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    preference_value: int
    diversity_count: int = Field(ge=0)
    travel_minutes: int = Field(ge=0)
    walking_meters: int = Field(ge=0)
    known_cost: Decimal = Field(ge=0)
    proximity_score: int = Field(default=0, ge=0)


class OptimizationSolution(BaseModel):
    model_config = ConfigDict(frozen=True)

    style: PlanStyle
    days: tuple[OptimizationDayAssignment, ...]
    objective_value: float
    objective_breakdown: ObjectiveBreakdown


class OptimizationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: OptimizationSolveStatus
    solver: str
    solutions: tuple[OptimizationSolution, ...]
    elapsed_ms: float = Field(ge=0)
    search_states: int = Field(ge=0)
    degraded_reason: str | None = None

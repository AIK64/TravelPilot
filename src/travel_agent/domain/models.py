from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.errors import PydanticUndefinedAnnotation


class Pace(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    INTENSIVE = "intensive"


class PlanStyle(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    EXPLORATION = "exploration"


class ItemType(StrEnum):
    ACTIVITY = "activity"
    MEAL = "meal"
    REST = "rest"
    TRANSPORT = "transport"


class ViolationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ValidationStatus(StrEnum):
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


class Coordinate(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)


class TimeWindow(BaseModel):
    start: time
    end: time

    @model_validator(mode="after")
    def validate_order(self) -> "TimeWindow":
        if self.end <= self.start:
            raise ValueError("time window end must be later than start")
        return self


class TransportAnchor(BaseModel):
    name: str
    at: datetime
    coordinate: Coordinate

    @model_validator(mode="after")
    def require_timezone(self) -> "TransportAnchor":
        if self.at.tzinfo is None or self.at.utcoffset() is None:
            raise ValueError("transport anchor datetime must be timezone-aware")
        return self


class LocationAnchor(BaseModel):
    name: str
    coordinate: Coordinate


class StayAnchorMode(StrEnum):
    PROVIDED = "provided"
    RECOMMENDED = "recommended"
    NOT_REQUIRED = "not_required"
    UNRESOLVED = "unresolved"


class StayAnchorResolution(BaseModel):
    """规划采用的住宿基点；与用户输入 accommodation 严格分离。"""

    mode: StayAnchorMode
    anchor: LocationAnchor | None = None
    confidence: float = Field(ge=0, le=1)
    confirmed: bool = False
    reference_poi_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


class DayBoundary(BaseModel):
    date: date
    start_role: str
    start_anchor: LocationAnchor
    end_role: str
    end_anchor: LocationAnchor


class MobilityConstraints(BaseModel):
    max_daily_walking_meters: int = Field(default=8_000, ge=0)
    max_daily_activity_minutes: int = Field(default=480, gt=0)
    needs_frequent_rest: bool = False


class TripSpec(BaseModel):
    destination: str = Field(min_length=1)
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=20)
    arrival: TransportAnchor
    departure: TransportAnchor
    accommodation: LocationAnchor | None = None
    total_budget: Decimal | None = Field(default=None, gt=0)
    interests: list[str] = Field(default_factory=list, max_length=100)
    avoid: list[str] = Field(default_factory=list, max_length=100)
    must_visit: list[str] = Field(default_factory=list, max_length=100)
    pace: Pace = Pace.BALANCED
    mobility: MobilityConstraints = Field(default_factory=MobilityConstraints)
    daily_start: time = time(9, 0)
    daily_end: time = time(20, 0)

    @field_validator("interests", "avoid", "must_visit")
    @classmethod
    def normalize_preference_terms(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]

    @model_validator(mode="after")
    def validate_trip_dates(self) -> "TripSpec":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be earlier than start_date")
        if self.arrival.at.date() < self.start_date:
            raise ValueError("arrival must not be earlier than start_date")
        if self.departure.at.date() > self.end_date:
            raise ValueError("departure must not be later than end_date")
        if self.departure.at <= self.arrival.at:
            raise ValueError("departure must be later than arrival")
        if self.daily_end <= self.daily_start:
            raise ValueError("daily_end must be later than daily_start")
        return self

    @property
    def day_count(self) -> int:
        return (self.end_date - self.start_date).days + 1


class POI(BaseModel):
    id: str
    name: str
    city: str
    coordinate: Coordinate
    categories: list[str]
    opening_window: TimeWindow
    estimated_duration_minutes: int = Field(gt=0)
    estimated_cost: Decimal = Field(default=Decimal("0"), ge=0)
    indoor_outdoor: str = "unknown"
    suitability_tags: list[str] = Field(default_factory=list)
    data_confidence: float = Field(default=1.0, ge=0, le=1)
    source: str = "mock"


class PlanningAssumption(BaseModel):
    field: str
    value: str
    reason: str
    source: "ValueSource"
    affected_dates: list[date] = Field(default_factory=list)
    policy_version: str = "v0.2-default-1"
    created_at: datetime


class PlanningPOI(BaseModel):
    facts: "POIFacts"
    opening_windows: dict[date, TimeWindow]
    duration_minutes: int
    party_cost: Decimal | None
    assumptions: list[PlanningAssumption] = Field(default_factory=list)
    field_sources: dict[str, "ValueSource"] = Field(default_factory=dict)
    opening_window_sources: dict[date, "ValueSource"] = Field(default_factory=dict)
    data_confidence: float = Field(ge=0, le=1)


class POIResolution(BaseModel):
    poi: PlanningPOI | None
    missing_fields: list[str] = Field(default_factory=list)


class POIResolutionIssue(BaseModel):
    poi_id: str
    poi_name: str
    missing_fields: list[str]
    required: bool = False


class PlanItem(BaseModel):
    # 一次性规划保持向后兼容；进入 v0.8 生命周期时由版本层补齐稳定 ID。
    item_id: str | None = None
    type: ItemType
    name: str
    start_at: datetime
    end_at: datetime
    poi_id: str | None = None
    coordinate: Coordinate | None = None
    travel_from_previous_minutes: int = Field(default=0, ge=0)
    distance_from_previous_meters: int = Field(default=0, ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    walking_distance_estimated: bool = False
    locked: bool = False

    @model_validator(mode="after")
    def validate_time_order(self) -> "PlanItem":
        if self.end_at <= self.start_at:
            raise ValueError("plan item end_at must be later than start_at")
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError("plan item datetimes must be timezone-aware")
        return self


class PlanRouteLeg(BaseModel):
    origin_name: str
    destination_name: str
    duration_minutes: int = Field(ge=0)
    distance_meters: int = Field(ge=0)
    mode: str
    provider: str
    data_confidence: float = Field(ge=0, le=1)
    walking_distance_meters: int = Field(default=0, ge=0)
    walking_distance_estimated: bool = False


class DayPlan(BaseModel):
    date: date
    theme: str
    primary_area: str
    items: list[PlanItem]
    known_estimated_cost: Decimal = Field(
        default=Decimal("0"),
        validation_alias=AliasChoices("known_estimated_cost", "estimated_cost"),
    )
    unknown_cost_item_count: int = Field(default=0, ge=0)
    total_travel_minutes: int = 0
    walking_distance_meters: int = 0
    fatigue_score: float = Field(default=0, ge=0, le=1)
    start_anchor: LocationAnchor | None = None
    end_anchor: LocationAnchor | None = None
    end_leg: PlanRouteLeg | None = None

    @computed_field
    @property
    def estimated_cost(self) -> Decimal:
        """保留 v0.1 响应字段，明确其仅表示已知费用。"""
        return self.known_estimated_cost


class PlanMetrics(BaseModel):
    preference_match: float = Field(ge=0, le=1)
    diversity: float = Field(ge=0, le=1)
    data_confidence: float = Field(ge=0, le=1)
    total_travel_minutes: int = Field(ge=0)
    walking_distance_meters: int = Field(ge=0)
    known_estimated_cost: Decimal = Field(
        ge=0,
        validation_alias=AliasChoices("known_estimated_cost", "estimated_cost"),
    )
    unknown_cost_item_count: int = Field(default=0, ge=0)
    fatigue_score: float = Field(ge=0, le=1)

    @computed_field
    @property
    def estimated_cost(self) -> Decimal:
        """保留 v0.1 响应字段，明确其仅表示已知费用。"""
        return self.known_estimated_cost


class Violation(BaseModel):
    type: str
    severity: ViolationSeverity
    message: str
    day: date | None = None
    entity_ids: list[str] = Field(default_factory=list)
    repair_hint: str | None = None


class ValidationResult(BaseModel):
    status: ValidationStatus
    violations: list[Violation] = Field(default_factory=list)

    @staticmethod
    def _status_from_violations(violations: list[Violation]) -> ValidationStatus:
        if any(item.severity is ViolationSeverity.ERROR for item in violations):
            return ValidationStatus.INVALID
        if violations:
            return ValidationStatus.VALID_WITH_WARNINGS
        return ValidationStatus.VALID

    @model_validator(mode="after")
    def validate_status(self) -> "ValidationResult":
        expected_status = self._status_from_violations(self.violations)
        if self.status is not expected_status:
            raise ValueError(
                "status must match the severity-derived result of violations"
            )
        return self

    @computed_field
    @property
    def valid(self) -> bool:
        """兼容旧响应，并让条件路由以 status 为准。"""
        return self.status is not ValidationStatus.INVALID

    @classmethod
    def from_violations(cls, violations: list[Violation]) -> "ValidationResult":
        return cls(
            status=cls._status_from_violations(violations),
            violations=violations,
        )


class PlanCandidate(BaseModel):
    id: str
    style: PlanStyle
    days: list[DayPlan]
    metrics: PlanMetrics
    validation: ValidationResult | None = None
    score: float | None = None
    reason_facts: list[str] = Field(default_factory=list)
    assumptions: list[PlanningAssumption] = Field(default_factory=list)


class PlanningRequest(BaseModel):
    trip: TripSpec
    max_replan_rounds: Annotated[int, Field(ge=0, le=5)] = 2


class PlanningResponse(BaseModel):
    status: str
    terminal_reason: str | None = None
    selected_plan: PlanCandidate | None
    candidates: list[PlanCandidate]
    iterations: int
    message: str | None = None
    critic_status: "CriticStatus" = Field(
        default="disabled", validate_default=True
    )
    critic_summary: "CriticExecutionSummary | None" = None
    candidate_critiques: list["SoftCritique"] = Field(default_factory=list)
    grounded_explanation: "GroundedExplanation | None" = None
    soft_iterations: int = Field(default=0, ge=0, le=1)
    stay_resolution: StayAnchorResolution | None = None
    agent_mode: str = "fixed_workflow"
    decision_count: int = Field(default=0, ge=0)
    action_summary: tuple[str, ...] = ()
    evidence_summary: dict[str, int] = Field(default_factory=dict)
    degraded_reasons: tuple[str, ...] = ()


def rebuild_provenance_models(
    *, poi_facts: type[BaseModel], value_source: type[StrEnum]
) -> None:
    """在 domain 层显式完成跨模块 provenance 类型的解析。"""
    types_namespace = {"POIFacts": poi_facts, "ValueSource": value_source}
    for model in (PlanningAssumption, PlanningPOI, POIResolution, PlanCandidate):
        model.model_rebuild(force=True, _types_namespace=types_namespace)
    _try_rebuild_critique_response_model()


def rebuild_critique_response_model(
    *,
    critic_status: type[StrEnum],
    execution_summary: type[BaseModel],
    soft_critique: type[BaseModel],
    grounded_explanation: type[BaseModel],
) -> None:
    try:
        PlanningResponse.model_rebuild(
            force=True,
            _types_namespace={
                "CriticStatus": critic_status,
                "CriticExecutionSummary": execution_summary,
                "SoftCritique": soft_critique,
                "GroundedExplanation": grounded_explanation,
            },
        )
    except PydanticUndefinedAnnotation:
        # tool_models 先导入时，ValueSource 尚未定义；其模块完成后会再次触发。
        return


def _try_rebuild_critique_response_model() -> None:
    try:
        from travel_agent.domain.critique_models import (
            CriticExecutionSummary,
            CriticStatus,
            GroundedExplanation,
            SoftCritique,
        )
    except ImportError:
        return
    rebuild_critique_response_model(
        critic_status=CriticStatus,
        execution_summary=CriticExecutionSummary,
        soft_critique=SoftCritique,
        grounded_explanation=GroundedExplanation,
    )


try:
    from travel_agent.domain.tool_models import POIFacts, ValueSource
except ImportError:
    # tool_models 先导入时会在其定义完成后调用 rebuild_provenance_models。
    pass
else:
    rebuild_provenance_models(poi_facts=POIFacts, value_source=ValueSource)


try:
    from travel_agent.domain.critique_models import (
        CriticExecutionSummary,
        CriticStatus,
        GroundedExplanation,
        SoftCritique,
    )
except ImportError:
    pass
else:
    rebuild_critique_response_model(
        critic_status=CriticStatus,
        execution_summary=CriticExecutionSummary,
        soft_critique=SoftCritique,
        grounded_explanation=GroundedExplanation,
    )

"""以显式、可追溯 policy 处理 POI 的未知事实。"""

from datetime import date, timedelta
from decimal import Decimal
import logging

from travel_agent.domain.models import (
    POIResolution,
    PlanningAssumption,
    PlanningPOI,
    TimeWindow,
    TripSpec,
)
from travel_agent.domain.tool_models import POIFacts, UnknownFactPolicy, ValueSource


_MODEL_TYPES = {"POIFacts": POIFacts, "ValueSource": ValueSource}
PlanningAssumption.model_rebuild(_types_namespace=_MODEL_TYPES)
PlanningPOI.model_rebuild(_types_namespace=_MODEL_TYPES)
POIResolution.model_rebuild(_types_namespace=_MODEL_TYPES)


DEFAULT_OPENING_WINDOW = TimeWindow(start="08:00", end="18:00")
DEFAULT_DURATION_MINUTES = 120
DEFAULT_COST_PER_PERSON = Decimal("45")
logger = logging.getLogger(__name__)
CONFIDENCE_DEDUCTION_PER_ASSUMPTION = 0.15
MINIMUM_CONFIDENCE = 0.1


class POIDefaultPolicy:
    """将 Provider 未返回的事实显式默认化，或在 strict 模式拒绝使用。"""

    def __init__(self, unknown_fact_policy: UnknownFactPolicy) -> None:
        self.unknown_fact_policy = unknown_fact_policy

    def resolve(self, facts: POIFacts, trip: TripSpec) -> POIResolution:
        """解析给定行程日期范围内的 POI 规划事实。"""
        opening_windows, opening_window_sources, default_opening_dates = self._resolve_opening_windows(
            facts, trip
        )
        missing_fields = [
            *( ["opening_window"] if default_opening_dates else [] ),
            *( ["duration_minutes"] if facts.suggested_duration_minutes is None else [] ),
            *( ["party_cost"] if facts.average_cost_per_person is None else [] ),
        ]
        if self.unknown_fact_policy is UnknownFactPolicy.STRICT and missing_fields:
            return POIResolution(poi=None, missing_fields=missing_fields)

        assumptions = self._build_assumptions(facts, missing_fields, default_opening_dates)
        duration_minutes = facts.suggested_duration_minutes or DEFAULT_DURATION_MINUTES
        party_cost = (
            facts.average_cost_per_person * trip.travelers
            if facts.average_cost_per_person is not None
            else DEFAULT_COST_PER_PERSON * trip.travelers
        )
        data_confidence = max(
            MINIMUM_CONFIDENCE,
            facts.data_confidence - CONFIDENCE_DEDUCTION_PER_ASSUMPTION * len(assumptions),
        )
        if missing_fields:
            logger.info("planning.poi_defaults poi_id=%s fields=%s duration_minutes=%s party_cost=%s",
                        facts.id, ",".join(missing_fields), duration_minutes, party_cost)
        return POIResolution(
            poi=PlanningPOI(
                facts=facts,
                opening_windows=opening_windows,
                duration_minutes=duration_minutes,
                party_cost=party_cost,
                assumptions=assumptions,
                field_sources={
                    "duration_minutes": (
                        ValueSource.PROVIDER
                        if facts.suggested_duration_minutes is not None
                        else ValueSource.DEFAULT
                    ),
                    "party_cost": (
                        ValueSource.DERIVED
                        if facts.average_cost_per_person is not None
                        else ValueSource.DEFAULT
                    ),
                    "data_confidence": ValueSource.DERIVED,
                },
                opening_window_sources=opening_window_sources,
                data_confidence=data_confidence,
            )
        )

    def _resolve_opening_windows(
        self, facts: POIFacts, trip: TripSpec
    ) -> tuple[dict[date, TimeWindow], dict[date, ValueSource], list[date]]:
        windows: dict[date, TimeWindow] = {}
        sources: dict[date, ValueSource] = {}
        default_dates: list[date] = []
        trip_date = trip.start_date
        while trip_date <= trip.end_date:
            if facts.today_opening_date == trip_date and facts.today_opening_window is not None:
                window = facts.today_opening_window
            else:
                window = facts.opening_windows_by_weekday.get(trip_date.weekday())
            if window is None:
                window = DEFAULT_OPENING_WINDOW
                source = ValueSource.DEFAULT
                default_dates.append(trip_date)
            else:
                source = ValueSource.PROVIDER
            windows[trip_date] = window
            sources[trip_date] = source
            trip_date += timedelta(days=1)
        return windows, sources, default_dates

    @staticmethod
    def _build_assumptions(
        facts: POIFacts, missing_fields: list[str], default_opening_dates: list[date]
    ) -> list[PlanningAssumption]:
        details = {
            "opening_window": (
                "08:00-18:00",
                "Provider 未提供适用于全部行程日期的营业时间，按默认 policy 补齐。",
            ),
            "duration_minutes": (
                str(DEFAULT_DURATION_MINUTES),
                "Provider 未提供建议游览时长，按默认 policy 估算。",
            ),
            "party_cost": (
                str(DEFAULT_COST_PER_PERSON),
                "Provider 未提供人均费用，按每人45元估算，规划费用乘以出行人数。",
            ),
        }
        return [
            PlanningAssumption(
                field=field,
                value=details[field][0],
                reason=details[field][1],
                source=ValueSource.DEFAULT,
                affected_dates=default_opening_dates if field == "opening_window" else [],
                created_at=facts.fetched_at,
            )
            for field in missing_fields
        ]

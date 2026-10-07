from __future__ import annotations

from datetime import datetime

from travel_agent.domain.models import PlanCandidate, TripSpec
from travel_agent.evidence.models import (
    EvidenceGap,
    EvidenceGapStatus,
    EvidenceKind,
    EvidenceRecord,
)


def derive_evidence_gaps(
    trip: TripSpec,
    evidence: tuple[EvidenceRecord, ...],
    *,
    candidate: PlanCandidate | None = None,
    now: datetime | None = None,
) -> tuple[EvidenceGap, ...]:
    active = tuple(item for item in evidence if item.active_at(now))

    def matching(kind: EvidenceKind, prefix: str) -> tuple[str, ...]:
        return tuple(
            item.evidence_id
            for item in active
            if item.kind is kind and item.subject_key.startswith(prefix)
        )

    poi_ids = matching(EvidenceKind.POI, "poi.candidates")
    route_ids = matching(EvidenceKind.ROUTE, "route.matrix")
    validation_ids = matching(EvidenceKind.VALIDATION, "candidate.validation")
    quality_ids = (
        matching(EvidenceKind.QUALITY, f"candidate.quality_review:{candidate.id}")
        if candidate is not None
        else ()
    )
    gaps = [
        EvidenceGap(
            key="trip.anchor.arrival",
            kind=EvidenceKind.USER_CONSTRAINT,
            required=True,
            status=EvidenceGapStatus.SATISFIED,
            reason_code="trip_spec_validated",
        ),
        EvidenceGap(
            key="trip.anchor.stay",
            kind=EvidenceKind.USER_CONSTRAINT,
            required=trip.day_count > 1,
            # 住宿不是硬必填项；缺省时由确定性 stay policy 推荐中心锚点，
            # 不应让 Planner 猜测酒店，也不应无条件打断用户。
            status=EvidenceGapStatus.SATISFIED,
            reason_code=(
                "stay_available"
                if trip.accommodation is not None or trip.day_count == 1
                else "recommended_stay_fallback_available"
            ),
        ),
        EvidenceGap(
            key="poi.candidates.interests",
            kind=EvidenceKind.POI,
            required=True,
            status=EvidenceGapStatus.SATISFIED if poi_ids else EvidenceGapStatus.OPEN,
            reason_code="evidence_available" if poi_ids else "poi_evidence_missing",
            satisfied_by=poi_ids,
        ),
        EvidenceGap(
            key="route.matrix.required",
            kind=EvidenceKind.ROUTE,
            required=True,
            status=(EvidenceGapStatus.SATISFIED if route_ids else EvidenceGapStatus.OPEN),
            reason_code="evidence_available" if route_ids else "route_evidence_missing",
            satisfied_by=route_ids,
        ),
        EvidenceGap(
            key="candidate.hard_validation",
            kind=EvidenceKind.VALIDATION,
            required=candidate is not None,
            status=(
                EvidenceGapStatus.SATISFIED
                if candidate is not None and validation_ids
                else EvidenceGapStatus.OPEN
            ),
            reason_code=(
                "validation_available"
                if candidate is not None and validation_ids
                else "validation_missing"
            ),
            satisfied_by=validation_ids,
        ),
        EvidenceGap(
            key="candidate.soft_quality",
            kind=EvidenceKind.QUALITY,
            required=candidate is not None,
            status=(
                EvidenceGapStatus.SATISFIED
                if candidate is not None and quality_ids
                else EvidenceGapStatus.OPEN
            ),
            reason_code=(
                "quality_review_available"
                if candidate is not None and quality_ids
                else "quality_review_missing"
            ),
            satisfied_by=quality_ids,
        ),
    ]
    return tuple(gaps)


def required_gaps_satisfied(gaps: tuple[EvidenceGap, ...]) -> bool:
    return all(
        not gap.required or gap.status is EvidenceGapStatus.SATISFIED for gap in gaps
    )

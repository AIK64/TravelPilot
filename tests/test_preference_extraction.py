from __future__ import annotations

import pytest

from travel_agent.domain.models import Pace
from travel_agent.identity.models import Principal
from travel_agent.memory.extraction import (
    PersistenceIntent,
    extract_preference_candidates,
    prepare_preference_evidence,
    validate_preference_candidates,
)
from travel_agent.memory.models import (
    MemoryCategory,
    MemoryProposalRequest,
    MemorySource,
)
from travel_agent.memory.repository import InMemoryPreferenceRepository
from travel_agent.memory.service import PreferenceMemoryService
from travel_agent.requirements.models import NaturalPlanningRequest, RequirementDraft


def test_explicit_long_term_preferences_become_validated_candidates() -> None:
    request = NaturalPlanningRequest(text="记住我以后旅行喜欢博物馆，节奏轻松")
    draft = RequirementDraft(interests=["博物馆"], pace=Pace.RELAXED)

    evidence = prepare_preference_evidence(request, draft)
    candidates = extract_preference_candidates(evidence)
    accepted, rejected = validate_preference_candidates(candidates)

    assert {item.category for item in accepted} == {
        MemoryCategory.PREFERRED_CATEGORIES,
        MemoryCategory.PACE,
    }
    assert all(
        item.persistence_intent is PersistenceIntent.EXPLICIT_LONG_TERM
        for item in accepted
    )
    assert all(item.source is MemorySource.EXPLICIT_USER for item in accepted)
    assert rejected == []


def test_current_trip_preferences_are_not_proposed_for_long_term_memory() -> None:
    request = NaturalPlanningRequest(text="这次旅行喜欢自然景点，安排轻松一点")
    draft = RequirementDraft(interests=["自然"], pace=Pace.RELAXED)

    candidates = extract_preference_candidates(
        prepare_preference_evidence(request, draft)
    )
    accepted, rejected = validate_preference_candidates(candidates)

    assert accepted == []
    assert len(rejected) == 2
    assert {item.reason_code for item in rejected} == {"current_trip_only"}


def test_walking_and_transport_preferences_are_normalized() -> None:
    request = NaturalPlanningRequest(
        text="以后旅行步行最多5公里，通常优先地铁"
    )

    evidence = prepare_preference_evidence(request, RequirementDraft())
    candidates, rejected = validate_preference_candidates(
        extract_preference_candidates(evidence)
    )

    assert rejected == []
    by_category = {item.category: item.value for item in candidates}
    assert by_category[MemoryCategory.WALKING_TOLERANCE] == 5_000
    assert by_category[MemoryCategory.PREFERRED_TRANSPORT] == ["subway"]


@pytest.mark.asyncio
async def test_pending_memory_proposal_is_idempotently_reused() -> None:
    service = PreferenceMemoryService(InMemoryPreferenceRepository())
    principal = Principal(tenant_id="tenant-a", user_id="user-a")
    request = MemoryProposalRequest(
        category=MemoryCategory.PACE,
        value="relaxed",
        source=MemorySource.MODEL_INFERENCE,
        confidence=0.72,
        reason="用户表达了偏好",
    )

    first = await service.propose(principal, request)
    second = await service.propose(principal, request)

    assert first.proposal_id == second.proposal_id

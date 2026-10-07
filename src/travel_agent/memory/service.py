from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import logging
from uuid import UUID

from travel_agent.domain.models import TripSpec
from travel_agent.execution.context import current_run_context
from travel_agent.execution.models import TraceEventType
from travel_agent.identity.models import Principal
from travel_agent.memory.context import PreferenceContextComposer
from travel_agent.memory.errors import MemoryConflictError, MemoryPolicyError
from travel_agent.memory.models import (
    AgentRole,
    ConfirmationStatus,
    MemoryProposal,
    MemoryProposalRequest,
    MemorySource,
    PersonalizationSettings,
    PersonalizationUpdateRequest,
    PreferenceContext,
    PreferenceCreateRequest,
    PreferenceExport,
    PreferenceList,
    PreferenceMemory,
    PreferenceUpdateRequest,
    ProposalStatus,
    utcnow,
)
from travel_agent.memory.policy import (
    normalize_preference_value,
    preference_content_hash,
)
from travel_agent.memory.repository import PreferenceRepository
from travel_agent.requirements.models import RequirementDraft


logger = logging.getLogger(__name__)


class PreferenceMemoryService:
    def __init__(
        self,
        repository: PreferenceRepository,
        *,
        context_max_tokens: int = 1_200,
        context_max_characters: int = 4_800,
    ) -> None:
        self.repository = repository
        self.composer = PreferenceContextComposer(
            max_tokens=context_max_tokens,
            max_characters=context_max_characters,
        )
        self._decisions: dict[tuple[str, str, str, str], object] = {}

    async def list(
        self, principal: Principal, *, include_inactive: bool = False
    ) -> PreferenceList:
        items = await self.repository.list_memories(
            principal.tenant_id,
            principal.user_id,
            include_inactive=include_inactive,
        )
        settings = await self.repository.get_personalization(
            principal.tenant_id, principal.user_id
        )
        return PreferenceList(items=items, personalization=settings)

    async def get(
        self, principal: Principal, memory_id: str
    ) -> PreferenceMemory:
        return await self.repository.get_memory(
            principal.tenant_id, principal.user_id, memory_id
        )

    async def create_explicit(
        self,
        principal: Principal,
        request: PreferenceCreateRequest,
        *,
        source_run_id: str | None = None,
    ) -> PreferenceMemory:
        value = normalize_preference_value(request.category, request.value)
        content_hash = preference_content_hash(
            category=request.category,
            value=value,
            scope=request.scope,
            scope_key=request.scope_key,
        )
        existing = await self.repository.find_content_hash(
            principal.tenant_id, principal.user_id, content_hash
        )
        if existing is not None:
            return existing
        now = utcnow()
        memory = PreferenceMemory(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            category=request.category,
            value=value,
            scope=request.scope,
            scope_key=request.scope_key,
            source=MemorySource.EXPLICIT_USER,
            source_run_id=source_run_id,
            confidence=1.0,
            confirmation_status=ConfirmationStatus.CONFIRMED,
            created_at=now,
            updated_at=now,
            expires_at=request.expires_at,
            content_hash=content_hash,
        )
        await self.repository.create_memory(memory)
        self._trace(
            TraceEventType.MEMORY_PERSISTED,
            "memory.create",
            memory_id=memory.memory_id,
            category=memory.category.value,
        )
        logger.info(
            "memory.created tenant=%s user=%s memory_id=%s category=%s",
            self._identity_hash(principal.tenant_id),
            self._identity_hash(principal.user_id),
            memory.memory_id,
            memory.category.value,
        )
        return memory

    async def update(
        self,
        principal: Principal,
        memory_id: str,
        request: PreferenceUpdateRequest,
    ) -> PreferenceMemory:
        current = await self.get(principal, memory_id)
        value = (
            normalize_preference_value(current.category, request.value)
            if request.value is not None
            else current.value
        )
        content_hash = preference_content_hash(
            category=current.category,
            value=value,
            scope=current.scope,
            scope_key=current.scope_key,
        )
        updated = current.model_copy(
            update={
                "value": value,
                "expires_at": request.expires_at,
                "revision": current.revision + 1,
                "updated_at": utcnow(),
                "content_hash": content_hash,
            }
        )
        await self.repository.save_memory(
            updated, expected_revision=request.expected_revision
        )
        return updated

    async def revoke(
        self,
        principal: Principal,
        memory_id: str,
        *,
        expected_revision: int,
    ) -> PreferenceMemory:
        current = await self.get(principal, memory_id)
        updated = current.model_copy(
            update={
                "revoked_at": utcnow(),
                "updated_at": utcnow(),
                "revision": current.revision + 1,
            }
        )
        await self.repository.save_memory(
            updated, expected_revision=expected_revision
        )
        self._trace(
            TraceEventType.MEMORY_REVOKED,
            "memory.revoke",
            memory_id=memory_id,
            category=current.category.value,
        )
        return updated

    async def delete(self, principal: Principal, memory_id: str) -> None:
        await self.repository.delete_memory(
            principal.tenant_id, principal.user_id, memory_id
        )
        logger.info(
            "memory.deleted tenant=%s user=%s memory_id=%s",
            self._identity_hash(principal.tenant_id),
            self._identity_hash(principal.user_id),
            memory_id,
        )

    async def clear(self, principal: Principal) -> int:
        return await self.repository.clear_memories(
            principal.tenant_id, principal.user_id
        )

    async def export(self, principal: Principal) -> PreferenceExport:
        listing = await self.list(principal, include_inactive=True)
        return PreferenceExport(
            items=listing.items,
            personalization=listing.personalization,
        )

    async def propose(
        self, principal: Principal, request: MemoryProposalRequest
    ) -> MemoryProposal:
        value = normalize_preference_value(request.category, request.value)
        content_hash = preference_content_hash(
            category=request.category,
            value=value,
            scope=request.scope,
            scope_key=request.scope_key,
        )
        pending = await self.repository.find_pending_proposal(
            principal.tenant_id, principal.user_id, content_hash
        )
        if pending is not None:
            return pending
        proposal = MemoryProposal(
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            category=request.category,
            value=value,
            scope=request.scope,
            scope_key=request.scope_key,
            source=request.source,
            source_run_id=request.source_run_id,
            confidence=request.confidence,
            reason=request.reason,
            expires_at=request.expires_at,
            content_hash=content_hash,
        )
        await self.repository.create_proposal(proposal)
        self._trace(
            TraceEventType.MEMORY_PROPOSAL_CREATED,
            "memory.propose",
            proposal_id=proposal.proposal_id,
            category=proposal.category.value,
        )
        return proposal

    async def confirm_proposal(
        self,
        principal: Principal,
        proposal_id: str,
        *,
        request_id: UUID,
    ) -> PreferenceMemory:
        decision_key = (
            principal.tenant_id,
            principal.user_id,
            proposal_id,
            str(request_id),
        )
        replay = self._decisions.get(decision_key)
        if isinstance(replay, PreferenceMemory):
            return replay
        proposal = await self.repository.get_proposal(
            principal.tenant_id, principal.user_id, proposal_id
        )
        if not proposal.pending_at():
            if proposal.status is ProposalStatus.CONFIRMED and proposal.memory_id:
                result = await self.get(principal, proposal.memory_id)
                self._decisions[decision_key] = result
                return result
            raise MemoryConflictError("proposal_not_pending")
        existing = await self.repository.find_content_hash(
            principal.tenant_id, principal.user_id, proposal.content_hash
        )
        now = utcnow()
        if existing is None:
            memory = PreferenceMemory(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                category=proposal.category,
                value=proposal.value,
                scope=proposal.scope,
                scope_key=proposal.scope_key,
                source=MemorySource.EXPLICIT_USER,
                source_run_id=proposal.source_run_id,
                confidence=1.0,
                confirmation_status=ConfirmationStatus.CONFIRMED,
                created_at=now,
                updated_at=now,
                expires_at=proposal.expires_at,
                content_hash=proposal.content_hash,
            )
            await self.repository.create_memory(memory)
        else:
            memory = existing
        resolved = proposal.model_copy(
            update={
                "status": ProposalStatus.CONFIRMED,
                "resolved_at": now,
                "memory_id": memory.memory_id,
            }
        )
        await self.repository.save_proposal(resolved)
        self._decisions[decision_key] = memory
        self._trace(
            TraceEventType.MEMORY_PERSISTED,
            "memory.confirm",
            memory_id=memory.memory_id,
            category=memory.category.value,
        )
        return memory

    async def reject_proposal(
        self,
        principal: Principal,
        proposal_id: str,
        *,
        request_id: UUID,
    ) -> MemoryProposal:
        decision_key = (
            principal.tenant_id,
            principal.user_id,
            proposal_id,
            str(request_id),
        )
        replay = self._decisions.get(decision_key)
        if isinstance(replay, MemoryProposal):
            return replay
        proposal = await self.repository.get_proposal(
            principal.tenant_id, principal.user_id, proposal_id
        )
        if proposal.status is ProposalStatus.REJECTED:
            self._decisions[decision_key] = proposal
            return proposal
        if not proposal.pending_at():
            raise MemoryConflictError("proposal_not_pending")
        resolved = proposal.model_copy(
            update={"status": ProposalStatus.REJECTED, "resolved_at": utcnow()}
        )
        await self.repository.save_proposal(resolved)
        self._decisions[decision_key] = resolved
        return resolved

    async def personalization(
        self, principal: Principal
    ) -> PersonalizationSettings:
        return await self.repository.get_personalization(
            principal.tenant_id, principal.user_id
        )

    async def update_personalization(
        self, principal: Principal, request: PersonalizationUpdateRequest
    ) -> PersonalizationSettings:
        current = await self.personalization(principal)
        if request.expected_revision is not None and (
            request.expected_revision != current.revision
        ):
            raise MemoryConflictError("stale_personalization_revision")
        updated = current.model_copy(
            update={
                "enabled": request.enabled,
                "revision": current.revision + 1,
                "updated_at": utcnow(),
            }
        )
        await self.repository.save_personalization(
            updated, expected_revision=current.revision
        )
        return updated

    async def context_for_trip(
        self,
        principal: Principal,
        *,
        trip: TripSpec,
        draft: RequirementDraft | None,
        agent_role: AgentRole,
    ) -> PreferenceContext:
        self._trace(
            TraceEventType.MEMORY_NAMESPACE_RESOLVED,
            "memory.namespace",
            namespace_hash=self._identity_hash(
                f"{principal.tenant_id}:{principal.user_id}"
            ),
        )
        settings = await self.personalization(principal)
        if not settings.enabled:
            return self.composer.compose(
                (), trip=trip, draft=draft, agent_role=agent_role
            )
        self._trace(
            TraceEventType.MEMORY_RETRIEVE_STARTED,
            "memory.retrieve",
            role=agent_role.value,
        )
        memories = await self.repository.list_memories(
            principal.tenant_id, principal.user_id
        )
        context = self.composer.compose(
            memories,
            trip=trip,
            draft=draft,
            agent_role=agent_role,
        )
        self._trace(
            TraceEventType.MEMORY_RETRIEVE_COMPLETED,
            "memory.retrieve",
            role=agent_role.value,
            retrieved_count=len(memories),
            selected_count=len(context.summaries),
        )
        if context.conflicts:
            self._trace(
                TraceEventType.MEMORY_CONFLICT_DETECTED,
                "memory.conflict",
                conflict_count=len(context.conflicts),
            )
        self._trace(
            TraceEventType.CONTEXT_COMPOSED,
            "context.compose",
            role=agent_role.value,
            selected_count=len(context.summaries),
            excluded_count=context.manifest.excluded_count,
            estimated_tokens=context.manifest.estimated_tokens,
            context_id=context.manifest.context_id,
        )
        logger.info(
            "memory.context.composed tenant=%s user=%s role=%s selected=%s "
            "excluded=%s estimated_tokens=%s",
            self._identity_hash(principal.tenant_id),
            self._identity_hash(principal.user_id),
            agent_role.value,
            len(context.summaries),
            context.manifest.excluded_count,
            context.manifest.estimated_tokens,
        )
        return context

    def apply_to_trip(
        self,
        trip: TripSpec,
        *,
        draft: RequirementDraft | None,
        context: PreferenceContext,
    ) -> TripSpec:
        return self.composer.apply_to_trip(trip, draft=draft, context=context)

    @staticmethod
    def _identity_hash(value: str) -> str:
        return sha256(value.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _trace(
        event_type: TraceEventType, operation: str, **attributes: str | int
    ) -> None:
        context = current_run_context()
        if context is None:
            return
        context.trace.record(
            event_type,
            status="completed",
            operation=operation,
            attributes=attributes,
        )

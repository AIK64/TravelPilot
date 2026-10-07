from __future__ import annotations

import json
from typing import Any

from travel_agent.domain.lifecycle_models import PlanSessionRecord, utcnow
from travel_agent.execution.errors import RunNotFoundError
from travel_agent.execution.models import AgentRunRecord, RunStatus, TraceEvent
from travel_agent.lifecycle.errors import LifecycleConflictError, LifecycleNotFoundError
from travel_agent.memory.errors import MemoryConflictError, MemoryNotFoundError
from travel_agent.memory.models import (
    MemoryProposal,
    PersonalizationSettings,
    PreferenceMemory,
)


async def create_postgres_pool(database_url: str):
    try:
        import asyncpg
    except ImportError as error:
        raise RuntimeError(
            "Install the production extra to use PostgreSQL repositories"
        ) from error
    return await asyncpg.create_pool(database_url, min_size=1, max_size=10)


def _payload(row: Any) -> object:
    value = row["payload"]
    return json.loads(value) if isinstance(value, str) else value


class PostgresPlanRepository:
    def __init__(self, pool) -> None:
        self.pool = pool

    async def create(self, session: PlanSessionRecord) -> None:
        try:
            await self.pool.execute(
                "INSERT INTO plan_sessions(session_id,tenant_id,user_id,revision,payload) "
                "VALUES($1,$2,$3,$4,$5::jsonb)",
                session.session_id,
                session.tenant_id,
                session.user_id,
                session.session_revision,
                session.model_dump_json(),
            )
        except Exception as error:
            if error.__class__.__name__ == "UniqueViolationError":
                raise LifecycleConflictError(session.session_id) from None
            raise

    async def get(self, session_id: str) -> PlanSessionRecord:
        row = await self.pool.fetchrow(
            "SELECT payload FROM plan_sessions WHERE session_id=$1", session_id
        )
        if row is None:
            raise LifecycleNotFoundError(session_id)
        return PlanSessionRecord.model_validate(_payload(row))

    async def save(
        self, session: PlanSessionRecord, *, expected_revision: int
    ) -> None:
        session.updated_at = utcnow()
        result = await self.pool.execute(
            "UPDATE plan_sessions SET revision=$1,payload=$2::jsonb,updated_at=now() "
            "WHERE session_id=$3 AND revision=$4",
            session.session_revision,
            session.model_dump_json(),
            session.session_id,
            expected_revision,
        )
        if result != "UPDATE 1":
            exists = await self.pool.fetchval(
                "SELECT 1 FROM plan_sessions WHERE session_id=$1", session.session_id
            )
            if exists is None:
                raise LifecycleNotFoundError(session.session_id)
            raise LifecycleConflictError(session.session_id, code="stale_revision")

    async def close(self) -> None:
        await self.pool.close()


class PostgresRunRepository:
    def __init__(self, pool) -> None:
        self.pool = pool

    async def create(self, run: AgentRunRecord) -> None:
        await self.pool.execute(
            "INSERT INTO agent_runs(run_id,tenant_id,user_id,status,kind,thread_id,"
            "session_id,request_id,started_at,payload) "
            "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)",
            run.run_id,
            run.tenant_id,
            run.user_id,
            run.status.value,
            run.run_kind.value,
            run.thread_id,
            run.session_id,
            run.request_id,
            run.started_at,
            run.model_dump_json(),
        )

    async def append_trace_event(self, event: TraceEvent) -> None:
        result = await self.pool.execute(
            "INSERT INTO trace_events(run_id,sequence,payload) "
            "SELECT $1,$2,$3::jsonb WHERE EXISTS "
            "(SELECT 1 FROM agent_runs WHERE run_id=$1) "
            "ON CONFLICT DO NOTHING",
            event.run_id,
            event.sequence,
            event.model_dump_json(),
        )
        if result == "INSERT 0 0":
            exists = await self.pool.fetchval(
                "SELECT 1 FROM agent_runs WHERE run_id=$1", event.run_id
            )
            if exists is None:
                raise RunNotFoundError(event.run_id)

    async def finalize(
        self, run: AgentRunRecord, events: tuple[TraceEvent, ...]
    ) -> None:
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                result = await connection.execute(
                    "UPDATE agent_runs SET status=$1,payload=$2::jsonb,ended_at=$3 "
                    "WHERE run_id=$4",
                    run.status.value,
                    run.model_dump_json(),
                    run.ended_at,
                    run.run_id,
                )
                if result != "UPDATE 1":
                    raise RunNotFoundError(run.run_id)
                if events:
                    await connection.executemany(
                        "INSERT INTO trace_events(run_id,sequence,payload) "
                        "VALUES($1,$2,$3::jsonb) ON CONFLICT DO NOTHING",
                        [
                            (run.run_id, event.sequence, event.model_dump_json())
                            for event in events
                        ],
                    )

    async def get(self, run_id: str) -> AgentRunRecord:
        row = await self.pool.fetchrow(
            "SELECT payload FROM agent_runs WHERE run_id=$1", run_id
        )
        if row is None:
            raise RunNotFoundError(run_id)
        return AgentRunRecord.model_validate(_payload(row))

    async def trace(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[TraceEvent, ...]:
        await self.get(run_id)
        rows = await self.pool.fetch(
            "SELECT payload FROM trace_events WHERE run_id=$1 AND sequence>$2 "
            "ORDER BY sequence LIMIT $3",
            run_id,
            after_sequence,
            limit,
        )
        return tuple(TraceEvent.model_validate(_payload(row)) for row in rows)

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("session_id", session_id, limit)

    async def list_for_thread(
        self, thread_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("thread_id", thread_id, limit)

    async def _list(
        self, field: str, value: str, limit: int
    ) -> tuple[AgentRunRecord, ...]:
        if field not in {"session_id", "thread_id"}:
            raise ValueError("unsupported run query field")
        rows = await self.pool.fetch(
            f"SELECT payload FROM agent_runs WHERE {field}=$1 "
            "ORDER BY started_at DESC LIMIT $2",
            value,
            limit,
        )
        return tuple(AgentRunRecord.model_validate(_payload(row)) for row in rows)

    async def find_request(
        self,
        *,
        request_id: str,
        session_id: str | None,
        thread_id: str | None,
    ) -> AgentRunRecord | None:
        row = await self.pool.fetchrow(
            "SELECT payload FROM agent_runs WHERE request_id=$1 AND status<>$2 "
            "AND ($3::text IS NULL OR session_id=$3) "
            "AND ($4::text IS NULL OR thread_id=$4) "
            "ORDER BY started_at DESC LIMIT 1",
            request_id,
            RunStatus.RUNNING.value,
            session_id,
            thread_id,
        )
        return AgentRunRecord.model_validate(_payload(row)) if row else None

    async def close(self) -> None:
        await self.pool.close()


class PostgresPreferenceRepository:
    def __init__(self, pool) -> None:
        self.pool = pool

    async def list_memories(
        self, tenant_id: str, user_id: str, *, include_inactive: bool = False
    ) -> tuple[PreferenceMemory, ...]:
        rows = await self.pool.fetch(
            "SELECT payload FROM preference_memories WHERE tenant_id=$1 AND user_id=$2 "
            + ("" if include_inactive else "AND revoked_at IS NULL ")
            + "ORDER BY updated_at DESC",
            tenant_id,
            user_id,
        )
        return tuple(PreferenceMemory.model_validate(_payload(row)) for row in rows)

    async def get_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> PreferenceMemory:
        row = await self.pool.fetchrow(
            "SELECT payload FROM preference_memories "
            "WHERE memory_id=$1 AND tenant_id=$2 AND user_id=$3",
            memory_id,
            tenant_id,
            user_id,
        )
        if row is None:
            raise MemoryNotFoundError(memory_id)
        return PreferenceMemory.model_validate(_payload(row))

    async def create_memory(self, memory: PreferenceMemory) -> None:
        try:
            await self.pool.execute(
                "INSERT INTO preference_memories(memory_id,tenant_id,user_id,content_hash,"
                "revision,updated_at,revoked_at,payload) VALUES($1,$2,$3,$4,$5,$6,$7,$8::jsonb)",
                memory.memory_id,
                memory.tenant_id,
                memory.user_id,
                memory.content_hash,
                memory.revision,
                memory.updated_at,
                memory.revoked_at,
                memory.model_dump_json(),
            )
        except Exception as error:
            if error.__class__.__name__ == "UniqueViolationError":
                raise MemoryConflictError("duplicate_memory") from None
            raise

    async def save_memory(
        self, memory: PreferenceMemory, *, expected_revision: int
    ) -> None:
        result = await self.pool.execute(
            "UPDATE preference_memories SET revision=$1,updated_at=$2,revoked_at=$3,"
            "payload=$4::jsonb WHERE memory_id=$5 AND tenant_id=$6 AND user_id=$7 "
            "AND revision=$8",
            memory.revision,
            memory.updated_at,
            memory.revoked_at,
            memory.model_dump_json(),
            memory.memory_id,
            memory.tenant_id,
            memory.user_id,
            expected_revision,
        )
        if result != "UPDATE 1":
            await self._memory_missing_or_conflict(memory)

    async def delete_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> None:
        result = await self.pool.execute(
            "DELETE FROM preference_memories WHERE memory_id=$1 AND tenant_id=$2 AND user_id=$3",
            memory_id,
            tenant_id,
            user_id,
        )
        if result != "DELETE 1":
            raise MemoryNotFoundError(memory_id)

    async def clear_memories(self, tenant_id: str, user_id: str) -> int:
        result = await self.pool.execute(
            "DELETE FROM preference_memories WHERE tenant_id=$1 AND user_id=$2",
            tenant_id,
            user_id,
        )
        return int(result.split()[-1])

    async def find_content_hash(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> PreferenceMemory | None:
        row = await self.pool.fetchrow(
            "SELECT payload FROM preference_memories WHERE tenant_id=$1 AND user_id=$2 "
            "AND content_hash=$3 AND revoked_at IS NULL",
            tenant_id,
            user_id,
            content_hash,
        )
        return PreferenceMemory.model_validate(_payload(row)) if row else None

    async def create_proposal(self, proposal: MemoryProposal) -> None:
        try:
            await self.pool.execute(
                "INSERT INTO memory_proposals(proposal_id,tenant_id,user_id,status,updated_at,payload) "
                "VALUES($1,$2,$3,$4,$5,$6::jsonb)",
                proposal.proposal_id,
                proposal.tenant_id,
                proposal.user_id,
                proposal.status.value,
                proposal.resolved_at or proposal.created_at,
                proposal.model_dump_json(),
            )
        except Exception as error:
            if error.__class__.__name__ == "UniqueViolationError":
                raise MemoryConflictError("duplicate_proposal") from None
            raise

    async def find_pending_proposal(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> MemoryProposal | None:
        rows = await self.pool.fetch(
            "SELECT payload FROM memory_proposals WHERE tenant_id=$1 AND user_id=$2 "
            "AND status='pending'",
            tenant_id,
            user_id,
        )
        for row in rows:
            item = MemoryProposal.model_validate(_payload(row))
            if item.content_hash == content_hash and item.pending_at():
                return item
        return None

    async def get_proposal(
        self, tenant_id: str, user_id: str, proposal_id: str
    ) -> MemoryProposal:
        row = await self.pool.fetchrow(
            "SELECT payload FROM memory_proposals WHERE proposal_id=$1 "
            "AND tenant_id=$2 AND user_id=$3",
            proposal_id,
            tenant_id,
            user_id,
        )
        if row is None:
            raise MemoryNotFoundError(proposal_id)
        return MemoryProposal.model_validate(_payload(row))

    async def save_proposal(self, proposal: MemoryProposal) -> None:
        result = await self.pool.execute(
            "UPDATE memory_proposals SET status=$1,updated_at=$2,payload=$3::jsonb "
            "WHERE proposal_id=$4 AND tenant_id=$5 AND user_id=$6",
            proposal.status.value,
            proposal.resolved_at or proposal.created_at,
            proposal.model_dump_json(),
            proposal.proposal_id,
            proposal.tenant_id,
            proposal.user_id,
        )
        if result != "UPDATE 1":
            raise MemoryNotFoundError(proposal.proposal_id)

    async def get_personalization(
        self, tenant_id: str, user_id: str
    ) -> PersonalizationSettings:
        row = await self.pool.fetchrow(
            "SELECT payload FROM personalization_settings WHERE tenant_id=$1 AND user_id=$2",
            tenant_id,
            user_id,
        )
        return (
            PersonalizationSettings.model_validate(_payload(row))
            if row
            else PersonalizationSettings(tenant_id=tenant_id, user_id=user_id)
        )

    async def save_personalization(
        self,
        settings: PersonalizationSettings,
        *,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is None:
            await self.pool.execute(
                "INSERT INTO personalization_settings(tenant_id,user_id,revision,payload) "
                "VALUES($1,$2,$3,$4::jsonb) ON CONFLICT(tenant_id,user_id) DO UPDATE "
                "SET revision=EXCLUDED.revision,payload=EXCLUDED.payload",
                settings.tenant_id,
                settings.user_id,
                settings.revision,
                settings.model_dump_json(),
            )
            return
        if expected_revision == 0:
            result = await self.pool.execute(
                "INSERT INTO personalization_settings(tenant_id,user_id,revision,payload) "
                "VALUES($1,$2,$3,$4::jsonb) ON CONFLICT(tenant_id,user_id) DO NOTHING",
                settings.tenant_id,
                settings.user_id,
                settings.revision,
                settings.model_dump_json(),
            )
            if result == "INSERT 0 1":
                return
        result = await self.pool.execute(
            "UPDATE personalization_settings SET revision=$1,payload=$2::jsonb "
            "WHERE tenant_id=$3 AND user_id=$4 AND revision=$5",
            settings.revision,
            settings.model_dump_json(),
            settings.tenant_id,
            settings.user_id,
            expected_revision,
        )
        if result != "UPDATE 1":
            raise MemoryConflictError("stale_personalization_revision")

    async def _memory_missing_or_conflict(self, memory: PreferenceMemory) -> None:
        exists = await self.pool.fetchval(
            "SELECT 1 FROM preference_memories WHERE memory_id=$1 AND tenant_id=$2 AND user_id=$3",
            memory.memory_id,
            memory.tenant_id,
            memory.user_id,
        )
        if exists is None:
            raise MemoryNotFoundError(memory.memory_id)
        raise MemoryConflictError("stale_memory_revision")

    async def close(self) -> None:
        await self.pool.close()

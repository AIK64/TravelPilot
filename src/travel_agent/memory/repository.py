from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
from typing import Protocol

from travel_agent.memory.errors import MemoryConflictError, MemoryNotFoundError
from travel_agent.memory.models import (
    MemoryProposal,
    PersonalizationSettings,
    PreferenceMemory,
)


class PreferenceRepository(Protocol):
    async def list_memories(
        self, tenant_id: str, user_id: str, *, include_inactive: bool = False
    ) -> tuple[PreferenceMemory, ...]: ...
    async def get_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> PreferenceMemory: ...
    async def create_memory(self, memory: PreferenceMemory) -> None: ...
    async def save_memory(
        self, memory: PreferenceMemory, *, expected_revision: int
    ) -> None: ...
    async def delete_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> None: ...
    async def clear_memories(self, tenant_id: str, user_id: str) -> int: ...
    async def find_content_hash(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> PreferenceMemory | None: ...
    async def create_proposal(self, proposal: MemoryProposal) -> None: ...
    async def find_pending_proposal(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> MemoryProposal | None: ...
    async def get_proposal(
        self, tenant_id: str, user_id: str, proposal_id: str
    ) -> MemoryProposal: ...
    async def save_proposal(self, proposal: MemoryProposal) -> None: ...
    async def get_personalization(
        self, tenant_id: str, user_id: str
    ) -> PersonalizationSettings: ...
    async def save_personalization(
        self,
        settings: PersonalizationSettings,
        *,
        expected_revision: int | None,
    ) -> None: ...
    async def close(self) -> None: ...


class InMemoryPreferenceRepository:
    def __init__(self) -> None:
        self._memories: dict[str, PreferenceMemory] = {}
        self._proposals: dict[str, MemoryProposal] = {}
        self._settings: dict[tuple[str, str], PersonalizationSettings] = {}
        self._lock = asyncio.Lock()

    async def list_memories(
        self, tenant_id: str, user_id: str, *, include_inactive: bool = False
    ) -> tuple[PreferenceMemory, ...]:
        async with self._lock:
            items = [
                item.model_copy(deep=True)
                for item in self._memories.values()
                if item.tenant_id == tenant_id
                and item.user_id == user_id
                and (include_inactive or item.active_at())
            ]
        items.sort(key=lambda item: (item.updated_at, item.memory_id), reverse=True)
        return tuple(items)

    async def get_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> PreferenceMemory:
        async with self._lock:
            item = self._memories.get(memory_id)
            if (
                item is None
                or item.tenant_id != tenant_id
                or item.user_id != user_id
            ):
                raise MemoryNotFoundError(memory_id)
            return item.model_copy(deep=True)

    async def create_memory(self, memory: PreferenceMemory) -> None:
        async with self._lock:
            if memory.memory_id in self._memories:
                raise MemoryConflictError("duplicate_memory_id")
            self._memories[memory.memory_id] = memory.model_copy(deep=True)

    async def save_memory(
        self, memory: PreferenceMemory, *, expected_revision: int
    ) -> None:
        async with self._lock:
            current = self._memories.get(memory.memory_id)
            if current is None:
                raise MemoryNotFoundError(memory.memory_id)
            if (
                current.tenant_id != memory.tenant_id
                or current.user_id != memory.user_id
            ):
                raise MemoryNotFoundError(memory.memory_id)
            if current.revision != expected_revision:
                raise MemoryConflictError("stale_memory_revision")
            self._memories[memory.memory_id] = memory.model_copy(deep=True)

    async def delete_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> None:
        await self.get_memory(tenant_id, user_id, memory_id)
        async with self._lock:
            del self._memories[memory_id]

    async def clear_memories(self, tenant_id: str, user_id: str) -> int:
        async with self._lock:
            ids = [
                memory_id
                for memory_id, item in self._memories.items()
                if item.tenant_id == tenant_id and item.user_id == user_id
            ]
            for memory_id in ids:
                del self._memories[memory_id]
            return len(ids)

    async def find_content_hash(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> PreferenceMemory | None:
        async with self._lock:
            for item in self._memories.values():
                if (
                    item.tenant_id == tenant_id
                    and item.user_id == user_id
                    and item.content_hash == content_hash
                    and item.revoked_at is None
                ):
                    return item.model_copy(deep=True)
        return None

    async def create_proposal(self, proposal: MemoryProposal) -> None:
        async with self._lock:
            if proposal.proposal_id in self._proposals:
                raise MemoryConflictError("duplicate_proposal_id")
            self._proposals[proposal.proposal_id] = proposal.model_copy(deep=True)

    async def find_pending_proposal(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> MemoryProposal | None:
        async with self._lock:
            for item in self._proposals.values():
                if (
                    item.tenant_id == tenant_id
                    and item.user_id == user_id
                    and item.content_hash == content_hash
                    and item.pending_at()
                ):
                    return item.model_copy(deep=True)
        return None

    async def get_proposal(
        self, tenant_id: str, user_id: str, proposal_id: str
    ) -> MemoryProposal:
        async with self._lock:
            item = self._proposals.get(proposal_id)
            if (
                item is None
                or item.tenant_id != tenant_id
                or item.user_id != user_id
            ):
                raise MemoryNotFoundError(proposal_id)
            return item.model_copy(deep=True)

    async def save_proposal(self, proposal: MemoryProposal) -> None:
        async with self._lock:
            current = self._proposals.get(proposal.proposal_id)
            if current is None:
                raise MemoryNotFoundError(proposal.proposal_id)
            if (
                current.tenant_id != proposal.tenant_id
                or current.user_id != proposal.user_id
            ):
                raise MemoryNotFoundError(proposal.proposal_id)
            self._proposals[proposal.proposal_id] = proposal.model_copy(deep=True)

    async def get_personalization(
        self, tenant_id: str, user_id: str
    ) -> PersonalizationSettings:
        async with self._lock:
            item = self._settings.get((tenant_id, user_id))
            if item is None:
                item = PersonalizationSettings(tenant_id=tenant_id, user_id=user_id)
                self._settings[(tenant_id, user_id)] = item
            return item.model_copy(deep=True)

    async def save_personalization(
        self,
        settings: PersonalizationSettings,
        *,
        expected_revision: int | None,
    ) -> None:
        key = (settings.tenant_id, settings.user_id)
        async with self._lock:
            current = self._settings.get(key)
            if current is not None and (
                expected_revision is None or current.revision != expected_revision
            ):
                raise MemoryConflictError("stale_personalization_revision")
            if current is None and expected_revision not in {None, 0}:
                raise MemoryConflictError("stale_personalization_revision")
            self._settings[key] = settings.model_copy(deep=True)

    async def close(self) -> None:
        return None


class SQLitePreferenceRepository:
    """单机持久化实现；所有权过滤和 CAS 均在 SQL 条件中完成。"""

    def __init__(self, path: str) -> None:
        database_path = Path(path).expanduser()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(database_path), check_same_thread=False
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS preference_memories ("
            "memory_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            "revision INTEGER NOT NULL, content_hash TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_preference_owner "
            "ON preference_memories(tenant_id,user_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_preference_hash "
            "ON preference_memories(tenant_id,user_id,content_hash)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS memory_proposals ("
            "proposal_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL,"
            "payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS personalization_settings ("
            "tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, revision INTEGER NOT NULL,"
            "payload TEXT NOT NULL, PRIMARY KEY(tenant_id,user_id))"
        )
        self._connection.commit()
        self._lock = asyncio.Lock()

    async def list_memories(
        self, tenant_id: str, user_id: str, *, include_inactive: bool = False
    ) -> tuple[PreferenceMemory, ...]:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM preference_memories "
                "WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            ).fetchall()
        items = [PreferenceMemory.model_validate_json(row[0]) for row in rows]
        if not include_inactive:
            items = [item for item in items if item.active_at()]
        items.sort(key=lambda item: (item.updated_at, item.memory_id), reverse=True)
        return tuple(items)

    async def get_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> PreferenceMemory:
        async with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM preference_memories "
                "WHERE memory_id=? AND tenant_id=? AND user_id=?",
                (memory_id, tenant_id, user_id),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(memory_id)
        return PreferenceMemory.model_validate_json(row[0])

    async def create_memory(self, memory: PreferenceMemory) -> None:
        async with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO preference_memories("
                    "memory_id,tenant_id,user_id,revision,content_hash,payload"
                    ") VALUES(?,?,?,?,?,?)",
                    (
                        memory.memory_id,
                        memory.tenant_id,
                        memory.user_id,
                        memory.revision,
                        memory.content_hash,
                        memory.model_dump_json(),
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError:
                raise MemoryConflictError("duplicate_memory_id") from None

    async def save_memory(
        self, memory: PreferenceMemory, *, expected_revision: int
    ) -> None:
        async with self._lock:
            cursor = self._connection.execute(
                "UPDATE preference_memories SET revision=?,content_hash=?,payload=? "
                "WHERE memory_id=? AND tenant_id=? AND user_id=? AND revision=?",
                (
                    memory.revision,
                    memory.content_hash,
                    memory.model_dump_json(),
                    memory.memory_id,
                    memory.tenant_id,
                    memory.user_id,
                    expected_revision,
                ),
            )
            self._connection.commit()
            if cursor.rowcount != 1:
                exists = self._connection.execute(
                    "SELECT 1 FROM preference_memories WHERE memory_id=? "
                    "AND tenant_id=? AND user_id=?",
                    (memory.memory_id, memory.tenant_id, memory.user_id),
                ).fetchone()
                if exists is None:
                    raise MemoryNotFoundError(memory.memory_id)
                raise MemoryConflictError("stale_memory_revision")

    async def delete_memory(
        self, tenant_id: str, user_id: str, memory_id: str
    ) -> None:
        async with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM preference_memories "
                "WHERE memory_id=? AND tenant_id=? AND user_id=?",
                (memory_id, tenant_id, user_id),
            )
            self._connection.commit()
        if cursor.rowcount != 1:
            raise MemoryNotFoundError(memory_id)

    async def clear_memories(self, tenant_id: str, user_id: str) -> int:
        async with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM preference_memories WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    async def find_content_hash(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> PreferenceMemory | None:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM preference_memories "
                "WHERE tenant_id=? AND user_id=? AND content_hash=?",
                (tenant_id, user_id, content_hash),
            ).fetchall()
        for row in rows:
            item = PreferenceMemory.model_validate_json(row[0])
            if item.revoked_at is None:
                return item
        return None

    async def create_proposal(self, proposal: MemoryProposal) -> None:
        async with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO memory_proposals("
                    "proposal_id,tenant_id,user_id,payload) VALUES(?,?,?,?)",
                    (
                        proposal.proposal_id,
                        proposal.tenant_id,
                        proposal.user_id,
                        proposal.model_dump_json(),
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError:
                raise MemoryConflictError("duplicate_proposal_id") from None

    async def find_pending_proposal(
        self, tenant_id: str, user_id: str, content_hash: str
    ) -> MemoryProposal | None:
        async with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM memory_proposals "
                "WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            ).fetchall()
        for row in rows:
            item = MemoryProposal.model_validate_json(row[0])
            if item.content_hash == content_hash and item.pending_at():
                return item
        return None

    async def get_proposal(
        self, tenant_id: str, user_id: str, proposal_id: str
    ) -> MemoryProposal:
        async with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM memory_proposals "
                "WHERE proposal_id=? AND tenant_id=? AND user_id=?",
                (proposal_id, tenant_id, user_id),
            ).fetchone()
        if row is None:
            raise MemoryNotFoundError(proposal_id)
        return MemoryProposal.model_validate_json(row[0])

    async def save_proposal(self, proposal: MemoryProposal) -> None:
        async with self._lock:
            cursor = self._connection.execute(
                "UPDATE memory_proposals SET payload=? "
                "WHERE proposal_id=? AND tenant_id=? AND user_id=?",
                (
                    proposal.model_dump_json(),
                    proposal.proposal_id,
                    proposal.tenant_id,
                    proposal.user_id,
                ),
            )
            self._connection.commit()
        if cursor.rowcount != 1:
            raise MemoryNotFoundError(proposal.proposal_id)

    async def get_personalization(
        self, tenant_id: str, user_id: str
    ) -> PersonalizationSettings:
        async with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM personalization_settings "
                "WHERE tenant_id=? AND user_id=?",
                (tenant_id, user_id),
            ).fetchone()
            if row is None:
                item = PersonalizationSettings(
                    tenant_id=tenant_id, user_id=user_id
                )
                self._connection.execute(
                    "INSERT INTO personalization_settings("
                    "tenant_id,user_id,revision,payload) VALUES(?,?,?,?)",
                    (tenant_id, user_id, item.revision, item.model_dump_json()),
                )
                self._connection.commit()
                return item
        return PersonalizationSettings.model_validate_json(row[0])

    async def save_personalization(
        self,
        settings: PersonalizationSettings,
        *,
        expected_revision: int | None,
    ) -> None:
        async with self._lock:
            row = self._connection.execute(
                "SELECT revision FROM personalization_settings "
                "WHERE tenant_id=? AND user_id=?",
                (settings.tenant_id, settings.user_id),
            ).fetchone()
            if row is None:
                if expected_revision not in {None, 0}:
                    raise MemoryConflictError("stale_personalization_revision")
                self._connection.execute(
                    "INSERT INTO personalization_settings("
                    "tenant_id,user_id,revision,payload) VALUES(?,?,?,?)",
                    (
                        settings.tenant_id,
                        settings.user_id,
                        settings.revision,
                        settings.model_dump_json(),
                    ),
                )
            else:
                if expected_revision is None or int(row[0]) != expected_revision:
                    raise MemoryConflictError("stale_personalization_revision")
                cursor = self._connection.execute(
                    "UPDATE personalization_settings SET revision=?,payload=? "
                    "WHERE tenant_id=? AND user_id=? AND revision=?",
                    (
                        settings.revision,
                        settings.model_dump_json(),
                        settings.tenant_id,
                        settings.user_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MemoryConflictError("stale_personalization_revision")
            self._connection.commit()

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()


@asynccontextmanager
async def open_preference_repository(
    *, backend: str, sqlite_path: str, database_url: str = ""
) -> AsyncIterator[PreferenceRepository]:
    repository: PreferenceRepository
    if backend == "postgres":
        from travel_agent.infrastructure.postgres import (
            PostgresPreferenceRepository,
            create_postgres_pool,
        )

        repository = PostgresPreferenceRepository(
            await create_postgres_pool(database_url)
        )
    elif backend == "sqlite":
        repository = SQLitePreferenceRepository(sqlite_path)
    else:
        repository = InMemoryPreferenceRepository()
    try:
        yield repository
    finally:
        await repository.close()

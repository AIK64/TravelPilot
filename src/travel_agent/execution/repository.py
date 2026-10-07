from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import sqlite3
from typing import Protocol

from travel_agent.execution.errors import RunNotFoundError
from travel_agent.execution.models import AgentRunRecord, RunStatus, TraceEvent


class RunRepository(Protocol):
    async def create(self, run: AgentRunRecord) -> None: ...
    async def append_trace_event(self, event: TraceEvent) -> None: ...
    async def finalize(
        self, run: AgentRunRecord, events: tuple[TraceEvent, ...]
    ) -> None: ...
    async def get(self, run_id: str) -> AgentRunRecord: ...
    async def trace(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[TraceEvent, ...]: ...
    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]: ...
    async def list_for_thread(
        self, thread_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]: ...
    async def find_request(
        self,
        *,
        request_id: str,
        session_id: str | None,
        thread_id: str | None,
    ) -> AgentRunRecord | None: ...
    async def close(self) -> None: ...


class InMemoryRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRunRecord] = {}
        self._events: dict[str, tuple[TraceEvent, ...]] = {}
        self._lock = asyncio.Lock()

    async def create(self, run: AgentRunRecord) -> None:
        async with self._lock:
            if run.run_id in self._runs:
                raise ValueError(f"duplicate run_id: {run.run_id}")
            self._runs[run.run_id] = run

    async def append_trace_event(self, event: TraceEvent) -> None:
        async with self._lock:
            if event.run_id not in self._runs:
                raise RunNotFoundError(event.run_id)
            values = self._events.get(event.run_id, ())
            if any(item.sequence == event.sequence for item in values):
                return
            self._events[event.run_id] = (*values, event)

    async def finalize(
        self, run: AgentRunRecord, events: tuple[TraceEvent, ...]
    ) -> None:
        async with self._lock:
            if run.run_id not in self._runs:
                raise RunNotFoundError(run.run_id)
            self._runs[run.run_id] = run
            self._events[run.run_id] = tuple(events)

    async def get(self, run_id: str) -> AgentRunRecord:
        async with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    async def trace(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[TraceEvent, ...]:
        await self.get(run_id)
        async with self._lock:
            values = self._events.get(run_id, ())
            return tuple(
                event for event in values if event.sequence > after_sequence
            )[:limit]

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("session_id", session_id, limit)

    async def list_for_thread(
        self, thread_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("thread_id", thread_id, limit)

    async def find_request(
        self,
        *,
        request_id: str,
        session_id: str | None,
        thread_id: str | None,
    ) -> AgentRunRecord | None:
        async with self._lock:
            values = tuple(self._runs.values())
        matches = [
            run
            for run in values
            if run.request_id == request_id
            and run.status is not RunStatus.RUNNING
            and (session_id is None or run.session_id == session_id)
            and (thread_id is None or run.thread_id == thread_id)
        ]
        return max(matches, key=lambda item: item.started_at) if matches else None

    async def close(self) -> None:
        return None

    async def _list(
        self, field: str, value: str, limit: int
    ) -> tuple[AgentRunRecord, ...]:
        async with self._lock:
            matches = [
                run for run in self._runs.values() if getattr(run, field) == value
            ]
        matches.sort(key=lambda item: item.started_at, reverse=True)
        return tuple(matches[:limit])


class SQLiteRunRepository:
    """单进程 SQLite Run Store；事件只追加，终态和事件在同一事务提交。"""

    def __init__(self, path: str) -> None:
        database_path = Path(path).expanduser()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(database_path), check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS agent_runs ("
            "run_id TEXT PRIMARY KEY, status TEXT NOT NULL, kind TEXT NOT NULL, "
            "thread_id TEXT, session_id TEXT, request_id TEXT, started_at TEXT NOT NULL, "
            "payload TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_runs_session "
            "ON agent_runs(session_id, started_at DESC)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_runs_thread "
            "ON agent_runs(thread_id, started_at DESC)"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS trace_events ("
            "run_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY(run_id, sequence))"
        )
        self._connection.commit()
        self._lock = asyncio.Lock()

    async def create(self, run: AgentRunRecord) -> None:
        async with self._lock:
            self._connection.execute(
                "INSERT INTO agent_runs(run_id,status,kind,thread_id,session_id,request_id,started_at,payload) "
                "VALUES (?,?,?,?,?,?,?,?)",
                self._row(run),
            )
            self._connection.commit()

    async def append_trace_event(self, event: TraceEvent) -> None:
        async with self._lock:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO trace_events(run_id,sequence,payload) "
                "SELECT ?,?,? WHERE EXISTS "
                "(SELECT 1 FROM agent_runs WHERE run_id=?)",
                (
                    event.run_id,
                    event.sequence,
                    event.model_dump_json(),
                    event.run_id,
                ),
            )
            self._connection.commit()
            if cursor.rowcount == 0:
                run_exists = self._connection.execute(
                    "SELECT 1 FROM agent_runs WHERE run_id=?", (event.run_id,)
                ).fetchone()
                if run_exists is None:
                    raise RunNotFoundError(event.run_id)

    async def finalize(
        self, run: AgentRunRecord, events: tuple[TraceEvent, ...]
    ) -> None:
        async with self._lock:
            try:
                self._connection.execute("BEGIN")
                cursor = self._connection.execute(
                    "UPDATE agent_runs SET status=?,kind=?,thread_id=?,session_id=?,"
                    "request_id=?,started_at=?,payload=? WHERE run_id=?",
                    (
                        run.status.value,
                        run.run_kind.value,
                        run.thread_id,
                        run.session_id,
                        run.request_id,
                        run.started_at.isoformat(),
                        run.model_dump_json(),
                        run.run_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RunNotFoundError(run.run_id)
                self._connection.executemany(
                    "INSERT OR IGNORE INTO trace_events(run_id,sequence,payload) VALUES (?,?,?)",
                    ((run.run_id, event.sequence, event.model_dump_json()) for event in events),
                )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    async def get(self, run_id: str) -> AgentRunRecord:
        async with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM agent_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return AgentRunRecord.model_validate_json(row[0])

    async def trace(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[TraceEvent, ...]:
        await self.get(run_id)
        async with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM trace_events WHERE run_id=? AND sequence>? "
                "ORDER BY sequence LIMIT ?",
                (run_id, after_sequence, limit),
            ).fetchall()
        return tuple(TraceEvent.model_validate_json(row[0]) for row in rows)

    async def list_for_session(
        self, session_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("session_id", session_id, limit)

    async def list_for_thread(
        self, thread_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        return await self._list("thread_id", thread_id, limit)

    async def find_request(
        self,
        *,
        request_id: str,
        session_id: str | None,
        thread_id: str | None,
    ) -> AgentRunRecord | None:
        clauses = ["request_id=?", "status<>'running'"]
        params: list[str | int] = [request_id]
        if session_id is not None:
            clauses.append("session_id=?")
            params.append(session_id)
        if thread_id is not None:
            clauses.append("thread_id=?")
            params.append(thread_id)
        params.append(1)
        async with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM agent_runs WHERE " + " AND ".join(clauses)
                + " ORDER BY started_at DESC LIMIT ?",
                tuple(params),
            ).fetchone()
        return AgentRunRecord.model_validate_json(row[0]) if row else None

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    async def _list(
        self, field: str, value: str, limit: int
    ) -> tuple[AgentRunRecord, ...]:
        if field not in {"session_id", "thread_id"}:
            raise ValueError("unsupported run query field")
        async with self._lock:
            rows = self._connection.execute(
                f"SELECT payload FROM agent_runs WHERE {field}=? "
                "ORDER BY started_at DESC LIMIT ?",
                (value, limit),
            ).fetchall()
        return tuple(AgentRunRecord.model_validate_json(row[0]) for row in rows)

    @staticmethod
    def _row(run: AgentRunRecord) -> tuple[str | None, ...]:
        return (
            run.run_id,
            run.status.value,
            run.run_kind.value,
            run.thread_id,
            run.session_id,
            run.request_id,
            run.started_at.isoformat(),
            run.model_dump_json(),
        )


@asynccontextmanager
async def open_run_repository(
    *, backend: str, sqlite_path: str, database_url: str = ""
) -> AsyncIterator[RunRepository]:
    repository: RunRepository
    if backend == "postgres":
        from travel_agent.infrastructure.postgres import (
            PostgresRunRepository,
            create_postgres_pool,
        )

        repository = PostgresRunRepository(await create_postgres_pool(database_url))
    elif backend == "sqlite":
        repository = SQLiteRunRepository(sqlite_path)
    else:
        repository = InMemoryRunRepository()
    try:
        yield repository
    finally:
        await repository.close()

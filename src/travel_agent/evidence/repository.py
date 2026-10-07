from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from travel_agent.evidence.models import EvidenceRecord


@runtime_checkable
class EvidenceRepository(Protocol):
    async def put(self, record: EvidenceRecord) -> EvidenceRecord: ...

    async def get(self, run_id: str, evidence_id: str) -> EvidenceRecord | None: ...

    async def list_for_run(self, run_id: str) -> tuple[EvidenceRecord, ...]: ...


class InMemoryEvidenceRepository:
    """按 Run 隔离的证据仓库；相同内容哈希幂等复用。"""

    def __init__(self, *, max_records_per_run: int = 256) -> None:
        self.max_records_per_run = max_records_per_run
        self._records: dict[str, dict[str, EvidenceRecord]] = {}
        self._hash_index: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def put(self, record: EvidenceRecord) -> EvidenceRecord:
        async with self._lock:
            hash_key = (record.run_id, record.content_hash)
            existing_id = self._hash_index.get(hash_key)
            if existing_id is not None:
                return self._records[record.run_id][existing_id]
            run_records = self._records.setdefault(record.run_id, {})
            if len(run_records) >= self.max_records_per_run:
                raise ValueError("max_evidence_records exceeded")
            run_records[record.evidence_id] = record
            self._hash_index[hash_key] = record.evidence_id
            return record

    async def get(self, run_id: str, evidence_id: str) -> EvidenceRecord | None:
        async with self._lock:
            return self._records.get(run_id, {}).get(evidence_id)

    async def list_for_run(self, run_id: str) -> tuple[EvidenceRecord, ...]:
        async with self._lock:
            return tuple(self._records.get(run_id, {}).values())

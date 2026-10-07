"""将经过 TraceRecorder 安全裁剪的事件追加到 UTF-8 文本文件。"""

import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from travel_agent.execution.models import TraceEvent


logger = logging.getLogger(__name__)

_SECRET_FIELDS = {
    "key", "apikey", "authorization", "headers", "cookie", "cookies",
    "password", "secret", "token", "accesstoken", "refreshtoken", "approvaltoken",
    "clientsecret", "credential", "credentials", "tenantid", "userid",
    "sig", "signature",
}
_SECRET_IN_TEXT = re.compile(
    r"(?i)([?&](?:key|api_key|token|access_token|signature|sig)=)[^&\s]+"
)
MAX_ACTION_IO_CHARS = 131072


def _is_secret_field(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    return normalized in _SECRET_FIELDS or normalized.endswith(("apikey", "password", "secret", "token"))


def safe_action_details(value: object, depth: int = 0) -> Any:
    """调试正文只进入文件；递归脱敏，不调用未知对象的 repr。"""
    if depth > 16:
        return {"truncated": True, "reason": "max_depth"}
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _is_secret_field(key)
            else safe_action_details(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [safe_action_details(item, depth + 1) for item in value]
    if isinstance(value, str):
        return _SECRET_IN_TEXT.sub(r"\1[REDACTED]", value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return {"unavailable_type": type(value).__name__}


def bounded_action_details(details: dict[str, object]) -> dict[str, object]:
    sanitized = safe_action_details(details)
    encoded = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= MAX_ACTION_IO_CHARS:
        return sanitized
    return {
        "truncated": True,
        "original_chars": len(encoded),
        "preview": encoded[:MAX_ACTION_IO_CHARS],
    }


class FileTraceSink:
    def __init__(self, directory: str | Path, run_id: str) -> None:
        # run_id 不能成为路径；非法或过长标识使用稳定摘要作为文件名。
        filename = (
            run_id
            if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id)
            else hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        )
        self.path = Path(directory) / f"{filename}.log"
        self._started = False

    def append(self, event: TraceEvent) -> None:
        """由 LiveTraceWriter 的串行队列在工作线程调用，每行一个完整事件。"""
        self._append_line(event.run_id, event.model_dump_json())

    def append_details(self, event: TraceEvent, details: dict[str, object]) -> None:
        record = {
            "record_type": "action_io",
            "schema_version": "trace-action-io-v1",
            "run_id": event.run_id,
            "trace_event_id": event.event_id,
            "trace_sequence": event.sequence,
            "parent_event_id": event.parent_event_id,
            "timestamp": event.timestamp.isoformat(),
            "operation": event.operation,
            "status": event.status,
            "details": details,
        }
        self._append_line(event.run_id, json.dumps(record, ensure_ascii=False, separators=(",", ":")))

    def _append_line(self, run_id: str, line: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line + "\n")
        if not self._started:
            self._started = True
            logger.info("agent_run.trace_file_created | run_id=%s path=%s", run_id, self.path)

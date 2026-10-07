from __future__ import annotations

from enum import StrEnum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field

from travel_agent.agents.actions import AgentPhase
from travel_agent.memory.models import AgentRole


class ToolRisk(StrEnum):
    READ_ONLY = "read_only"
    USER_VISIBLE = "user_visible"


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    input_schema_name: str = Field(min_length=1, max_length=120)
    output_schema_name: str = Field(min_length=1, max_length=120)
    input_schema: dict[str, object] = Field(default_factory=dict)
    input_example: dict[str, object] = Field(default_factory=dict)
    allowed_roles: frozenset[AgentRole]
    allowed_phases: frozenset[AgentPhase]
    risk: ToolRisk
    max_batch_size: int = Field(ge=1, le=1_000)
    cacheable: bool
    estimated_cost_units: int = Field(ge=0, le=10_000)
    forbidden_argument_names: frozenset[str] = frozenset(
        {"tenant_id", "user_id", "provider", "api_key", "url"}
    )


class ToolRegistry:
    def __init__(self, descriptors: Iterable[ToolDescriptor]) -> None:
        items = tuple(descriptors)
        self._items = {item.name: item for item in items}
        if len(self._items) != len(items):
            raise ValueError("tool descriptor names must be unique")

    def get(self, name: str) -> ToolDescriptor | None:
        return self._items.get(name)

    def manifest(
        self, *, role: AgentRole, phase: AgentPhase
    ) -> tuple[ToolDescriptor, ...]:
        # 延迟导入避免依赖环；参数 Schema 直接来自 Executor 的执行模型。
        from travel_agent.tools.agent_executor import TOOL_INPUT_MODELS

        examples = {
            "poi.search": {"city": "泉州", "keywords": ["景点"], "limit": 12},
            "route.build_matrix": {"mode": "driving", "strategy": 32},
            "anchor.resolve": {"roles": ["arrival", "departure", "accommodation"]},
            "weather.snapshot": {"destination": "泉州", "start_date": "2026-10-02", "end_date": "2026-10-05"},
        }
        return tuple(
            item.model_copy(update={
                "input_schema": TOOL_INPUT_MODELS[item.name].model_json_schema(),
                "input_example": examples.get(item.name, {}),
            }) if item.name in TOOL_INPUT_MODELS else item
            for item in self._items.values()
            if role in item.allowed_roles and phase in item.allowed_phases
        )


def default_agent_tool_registry() -> ToolRegistry:
    planner = frozenset({AgentRole.PLANNER})
    replanner = frozenset({AgentRole.REPLANNER})
    research = frozenset({AgentPhase.RESEARCHING})
    repair = frozenset({AgentPhase.INVALID, AgentPhase.REPAIRING})
    return ToolRegistry(
        (
            ToolDescriptor(name="poi.search", description="搜索标准化 POI 候选", input_schema_name="POISearchToolInput", output_schema_name="POIEvidence", allowed_roles=planner | replanner, allowed_phases=research | repair, risk=ToolRisk.READ_ONLY, max_batch_size=12, cacheable=True, estimated_cost_units=1),
            ToolDescriptor(name="anchor.resolve", description="解析抵离或住宿锚点", input_schema_name="AnchorResolveToolInput", output_schema_name="AnchorEvidence", allowed_roles=planner, allowed_phases=research, risk=ToolRisk.READ_ONLY, max_batch_size=3, cacheable=True, estimated_cost_units=1),
            ToolDescriptor(name="route.build_matrix", description="按服务端规划策略批量构建完整候选路线矩阵", input_schema_name="RouteMatrixToolInput", output_schema_name="RouteEvidence", allowed_roles=planner, allowed_phases=research, risk=ToolRisk.READ_ONLY, max_batch_size=256, cacheable=True, estimated_cost_units=4),
            ToolDescriptor(name="route.load_delta", description="加载局部修复所需路线增量", input_schema_name="RouteDeltaToolInput", output_schema_name="RouteEvidence", allowed_roles=replanner, allowed_phases=repair, risk=ToolRisk.READ_ONLY, max_batch_size=64, cacheable=True, estimated_cost_units=2),
            ToolDescriptor(name="weather.snapshot", description="读取行程日期范围天气风险摘要", input_schema_name="WeatherSnapshotToolInput", output_schema_name="WeatherEvidence", allowed_roles=planner, allowed_phases=research, risk=ToolRisk.READ_ONLY, max_batch_size=1, cacheable=True, estimated_cost_units=1),
            ToolDescriptor(name="preference.retrieve", description="读取已确认且作用域匹配的偏好摘要", input_schema_name="PreferenceRetrieveToolInput", output_schema_name="PreferenceEvidence", allowed_roles=planner | replanner, allowed_phases=research | repair, risk=ToolRisk.READ_ONLY, max_batch_size=20, cacheable=False, estimated_cost_units=0),
        )
    )

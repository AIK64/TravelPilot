from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping
from urllib.parse import urlsplit

from travel_agent.domain.optimization_models import OptimizationBudget
from travel_agent.domain.tool_models import ProviderMode, UnknownFactPolicy
from travel_agent.execution.models import ExecutionBudget
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.requirements.models import RequirementProviderMode


class CheckpointBackend(StrEnum):
    MEMORY = "memory"
    SQLITE = "sqlite"


class RunStoreBackend(StrEnum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


class CriticProviderMode(StrEnum):
    DISABLED = "disabled"
    MOCK = "mock"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class EditProviderMode(StrEnum):
    MOCK = "mock"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class WeatherProviderMode(StrEnum):
    MOCK = "mock"
    AMAP = "amap"
    QWEATHER = "qweather"


class AgentMode(StrEnum):
    FIXED_WORKFLOW = "fixed_workflow"
    DYNAMIC_PLANNER = "dynamic_planner"
    SHADOW_DYNAMIC_PLANNER = "shadow_dynamic_planner"
    SINGLE_GRAPH = "single_graph"
    SPECIALIST_SUBAGENTS = "specialist_subagents"
    SHADOW_SUBAGENTS = "shadow_subagents"


class PlannerProviderMode(StrEnum):
    MOCK = "mock"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class ReplannerProviderMode(StrEnum):
    DISABLED = "disabled"
    DETERMINISTIC = "deterministic"
    MOCK = "mock"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class AsyncExecutionBackend(StrEnum):
    LOCAL = "local"
    REDIS = "redis"


@dataclass(frozen=True, slots=True)
class Settings:
    provider: ProviderMode = ProviderMode.MOCK
    amap_api_key: str | None = field(default=None, repr=False)
    baidu_api_key: str | None = field(default=None, repr=False)
    map_fallback_provider: str = "none"
    tool_timeout_seconds: float = 5.0
    tool_max_attempts: int = 3
    tool_backoff_base_seconds: float = 0.25
    tool_max_backoff_seconds: float = 2.0
    tool_max_concurrency: int = 5
    tool_cache_max_entries: int = 2048
    poi_cache_ttl_seconds: int = 3600
    route_cache_ttl_seconds: int = 300
    poi_query_limit: int = 10
    poi_candidate_limit: int = 12
    poi_max_queries: int = 12
    unknown_fact_policy: UnknownFactPolicy = UnknownFactPolicy.ASSUME_WITH_WARNING
    amap_driving_strategy: int = 32
    max_walking_leg_meters: int = 1_500
    use_real_walking_routes: bool = True
    optimization_max_solve_ms: int = 800
    optimization_max_search_states: int = 20_000
    optimization_candidate_limit: int = 8
    optimization_variant_count: int = 3
    critic_provider: CriticProviderMode = CriticProviderMode.MOCK
    critic_model: str = "mock-soft-critic-v1"
    critic_timeout_seconds: float = 20.0
    critic_max_attempts: int = 2
    critic_grounding_max_attempts: int = 2
    critic_max_input_chars: int = 24_000
    critic_max_output_tokens: int = 4_096
    critic_quality_threshold: float = 70.0
    critic_min_improvement: float = 5.0
    max_soft_replan_rounds: int = 1
    requirement_provider: RequirementProviderMode = RequirementProviderMode.MOCK
    openai_api_key: str | None = field(default=None, repr=False)
    requirement_model: str = "mock-requirement-v1"
    requirement_timeout_seconds: float = 20.0
    requirement_max_attempts: int = 2
    requirement_backoff_base_seconds: float = 0.5
    requirement_max_backoff_seconds: float = 2.0
    edit_provider: EditProviderMode = EditProviderMode.MOCK
    edit_model: str = "mock-plan-edit-v1"
    edit_timeout_seconds: float = 20.0
    edit_max_attempts: int = 2
    edit_max_output_tokens: int = 1_200
    plan_max_versions: int = 20
    plan_max_affected_days: int = 2
    weather_provider: WeatherProviderMode = WeatherProviderMode.MOCK
    qweather_api_host: str = ""
    qweather_token: str | None = field(default=None, repr=False)
    weather_fallback_provider: str = "none"
    weather_cache_ttl_seconds: int = 1_800
    weather_stale_max_seconds: int = 21_600
    weather_max_events: int = 50
    weather_max_poi_searches: int = 4
    weather_max_alternatives: int = 6
    weather_exposure_min_confidence: float = 0.8
    deepseek_api_key: str | None = field(default=None, repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = ""
    checkpoint_backend: CheckpointBackend = CheckpointBackend.MEMORY
    checkpoint_sqlite_path: str = ".data/travel-agent-checkpoints.sqlite3"
    plan_sqlite_path: str = ".data/travel-agent-plans.sqlite3"
    plan_store_backend: RunStoreBackend = RunStoreBackend.MEMORY
    run_store_backend: RunStoreBackend = RunStoreBackend.MEMORY
    run_sqlite_path: str = ".data/travel-agent-runs.sqlite3"
    memory_store_backend: RunStoreBackend = RunStoreBackend.MEMORY
    memory_sqlite_path: str = ".data/travel-agent-memory.sqlite3"
    database_url: str = ""
    memory_context_max_tokens: int = 1_200
    memory_context_max_characters: int = 4_800
    agent_mode: AgentMode = AgentMode.DYNAMIC_PLANNER
    agent_max_handoffs: int = 8
    planner_provider: PlannerProviderMode = PlannerProviderMode.MOCK
    planner_model: str = "mock-dynamic-planner-v1"
    planner_timeout_seconds: float = 20.0
    planner_max_attempts: int = 2
    planner_max_output_tokens: int = 1_200
    planner_max_decisions: int = 12
    planner_max_invalid_actions: int = 2
    planner_max_repeated_actions: int = 2
    replanner_provider: ReplannerProviderMode = ReplannerProviderMode.DETERMINISTIC
    replanner_model: str = "deterministic-replanner-v1"
    replanner_timeout_seconds: float = 20.0
    replanner_max_attempts: int = 2
    replanner_max_output_tokens: int = 1_600
    max_evidence_records: int = 256
    max_observation_history: int = 8
    max_planner_context_tokens: int = 6_000
    max_replanner_context_tokens: int = 5_000
    dev_identity_enabled: bool = True
    dev_tenant_id: str = "local"
    dev_user_id: str = "demo"
    mcp_allowed_hosts: tuple[str, ...] = (
        "127.0.0.1:*",
        "localhost:*",
        "[::1]:*",
        "testserver",
        "api:*",
    )
    mcp_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    )
    async_execution_backend: AsyncExecutionBackend = AsyncExecutionBackend.LOCAL
    redis_url: str = "redis://localhost:6379/0"
    run_budget_profile: str = "default-v1"
    run_max_graph_steps: int = 120
    run_max_tool_calls: int = 160
    run_max_provider_attempts: int = 240
    run_max_llm_calls: int = 8
    run_max_llm_attempts: int = 12
    run_max_llm_input_chars: int = 80_000
    run_max_input_tokens: int = 40_000
    run_max_output_tokens: int = 12_000
    run_max_repair_rounds: int = 4
    run_max_interrupts: int = 1
    run_max_checkpoint_writes: int = 160
    run_max_trace_events: int = 512
    run_max_repeated_fingerprint_count: int = 2
    run_deadline_seconds: float = 120.0
    trace_attribute_max_chars: int = 256
    trace_file_enabled: bool = True
    trace_file_dir: str = ".data/traces"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        settings = cls(
            provider=ProviderMode(source.get("TRAVEL_PROVIDER", "mock").strip().lower()),
            amap_api_key=source.get("AMAP_API_KEY", "").strip() or None,
            baidu_api_key=source.get("BAIDU_MAP_AK", "").strip() or None,
            map_fallback_provider=source.get(
                "MAP_FALLBACK_PROVIDER", "none"
            ).strip().lower(),
            tool_timeout_seconds=float(source.get("TOOL_TIMEOUT_SECONDS", "5")),
            tool_max_attempts=int(source.get("TOOL_MAX_ATTEMPTS", "3")),
            tool_backoff_base_seconds=float(
                source.get("TOOL_BACKOFF_BASE_SECONDS", "0.25")
            ),
            tool_max_backoff_seconds=float(
                source.get("TOOL_MAX_BACKOFF_SECONDS", "2")
            ),
            tool_max_concurrency=int(source.get("TOOL_MAX_CONCURRENCY", "5")),
            tool_cache_max_entries=int(source.get("TOOL_CACHE_MAX_ENTRIES", "2048")),
            poi_cache_ttl_seconds=int(source.get("POI_CACHE_TTL_SECONDS", "3600")),
            route_cache_ttl_seconds=int(source.get("ROUTE_CACHE_TTL_SECONDS", "300")),
            poi_query_limit=int(source.get("POI_QUERY_LIMIT", "10")),
            poi_candidate_limit=int(source.get("POI_CANDIDATE_LIMIT", "12")),
            poi_max_queries=int(source.get("POI_MAX_QUERIES", "12")),
            unknown_fact_policy=UnknownFactPolicy(
                source.get("UNKNOWN_FACT_POLICY", "assume_with_warning").strip().lower()
            ),
            amap_driving_strategy=int(source.get("AMAP_DRIVING_STRATEGY", "32")),
            max_walking_leg_meters=int(
                source.get("MAX_WALKING_LEG_METERS", "1500")
            ),
            use_real_walking_routes=_parse_bool(
                source.get("USE_REAL_WALKING_ROUTES", "true"),
                name="USE_REAL_WALKING_ROUTES",
            ),
            optimization_max_solve_ms=int(
                source.get("OPTIMIZATION_MAX_SOLVE_MS", "800")
            ),
            optimization_max_search_states=int(
                source.get("OPTIMIZATION_MAX_SEARCH_STATES", "20000")
            ),
            optimization_candidate_limit=int(
                source.get("OPTIMIZATION_CANDIDATE_LIMIT", "8")
            ),
            optimization_variant_count=int(
                source.get("OPTIMIZATION_VARIANT_COUNT", "3")
            ),
            critic_provider=CriticProviderMode(
                source.get("CRITIC_PROVIDER", "mock").strip().lower()
            ),
            critic_model=source.get(
                "CRITIC_MODEL", "mock-soft-critic-v1"
            ).strip(),
            critic_timeout_seconds=float(
                source.get("CRITIC_TIMEOUT_SECONDS", "20")
            ),
            critic_max_attempts=int(source.get("CRITIC_MAX_ATTEMPTS", "2")),
            critic_grounding_max_attempts=int(
                source.get("CRITIC_GROUNDING_MAX_ATTEMPTS", "2")
            ),
            critic_max_input_chars=int(
                source.get("CRITIC_MAX_INPUT_CHARS", "24000")
            ),
            critic_max_output_tokens=int(
                source.get("CRITIC_MAX_OUTPUT_TOKENS", "4096")
            ),
            critic_quality_threshold=float(
                source.get("CRITIC_QUALITY_THRESHOLD", "70")
            ),
            critic_min_improvement=float(
                source.get("CRITIC_MIN_IMPROVEMENT", "5")
            ),
            max_soft_replan_rounds=int(
                source.get("MAX_SOFT_REPLAN_ROUNDS", "1")
            ),
            requirement_provider=RequirementProviderMode(
                source.get("REQUIREMENT_PROVIDER", "mock").strip().lower()
            ),
            openai_api_key=source.get("OPENAI_API_KEY", "").strip() or None,
            requirement_model=source.get(
                "REQUIREMENT_MODEL", "mock-requirement-v1"
            ).strip(),
            requirement_timeout_seconds=float(
                source.get("REQUIREMENT_TIMEOUT_SECONDS", "20")
            ),
            requirement_max_attempts=int(
                source.get("REQUIREMENT_MAX_ATTEMPTS", "2")
            ),
            requirement_backoff_base_seconds=float(
                source.get("REQUIREMENT_BACKOFF_BASE_SECONDS", "0.5")
            ),
            requirement_max_backoff_seconds=float(
                source.get("REQUIREMENT_MAX_BACKOFF_SECONDS", "2")
            ),
            edit_provider=EditProviderMode(
                source.get("EDIT_PROVIDER", "mock").strip().lower()
            ),
            edit_model=source.get("EDIT_MODEL", "mock-plan-edit-v1").strip(),
            edit_timeout_seconds=float(source.get("EDIT_TIMEOUT_SECONDS", "20")),
            edit_max_attempts=int(source.get("EDIT_MAX_ATTEMPTS", "2")),
            edit_max_output_tokens=int(
                source.get("EDIT_MAX_OUTPUT_TOKENS", "1200")
            ),
            plan_max_versions=int(source.get("PLAN_MAX_VERSIONS", "20")),
            plan_max_affected_days=int(
                source.get("PLAN_MAX_AFFECTED_DAYS", "2")
            ),
            weather_provider=WeatherProviderMode(
                source.get("WEATHER_PROVIDER", "mock").strip().lower()
            ),
            qweather_api_host=source.get("QWEATHER_API_HOST", "").strip().rstrip("/"),
            qweather_token=source.get("QWEATHER_TOKEN", "").strip() or None,
            weather_fallback_provider=source.get(
                "WEATHER_FALLBACK_PROVIDER", "none"
            ).strip().lower(),
            weather_cache_ttl_seconds=int(
                source.get("WEATHER_CACHE_TTL_SECONDS", "1800")
            ),
            weather_stale_max_seconds=int(
                source.get("WEATHER_STALE_MAX_SECONDS", "21600")
            ),
            weather_max_events=int(source.get("WEATHER_MAX_EVENTS", "50")),
            weather_max_poi_searches=int(
                source.get("WEATHER_MAX_POI_SEARCHES", "4")
            ),
            weather_max_alternatives=int(
                source.get("WEATHER_MAX_ALTERNATIVES", "6")
            ),
            weather_exposure_min_confidence=float(
                source.get("WEATHER_EXPOSURE_MIN_CONFIDENCE", "0.8")
            ),
            deepseek_api_key=source.get("DEEPSEEK_API_KEY", "").strip() or None,
            deepseek_base_url=source.get(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            )
            .strip()
            .rstrip("/"),
            deepseek_model=source.get("DEEPSEEK_MODEL", "").strip(),
            checkpoint_backend=CheckpointBackend(
                source.get("CHECKPOINT_BACKEND", "memory").strip().lower()
            ),
            checkpoint_sqlite_path=source.get(
                "CHECKPOINT_SQLITE_PATH",
                ".data/travel-agent-checkpoints.sqlite3",
            ).strip(),
            plan_sqlite_path=source.get(
                "PLAN_SQLITE_PATH", ".data/travel-agent-plans.sqlite3"
            ).strip(),
            plan_store_backend=RunStoreBackend(
                source.get(
                    "PLAN_STORE_BACKEND", source.get("CHECKPOINT_BACKEND", "memory")
                ).strip().lower()
            ),
            run_store_backend=RunStoreBackend(
                source.get("RUN_STORE_BACKEND", "memory").strip().lower()
            ),
            run_sqlite_path=source.get(
                "RUN_SQLITE_PATH", ".data/travel-agent-runs.sqlite3"
            ).strip(),
            memory_store_backend=RunStoreBackend(
                source.get("MEMORY_STORE_BACKEND", "memory").strip().lower()
            ),
            memory_sqlite_path=source.get(
                "MEMORY_SQLITE_PATH", ".data/travel-agent-memory.sqlite3"
            ).strip(),
            database_url=source.get("DATABASE_URL", "").strip(),
            memory_context_max_tokens=int(
                source.get("MEMORY_CONTEXT_MAX_TOKENS", "1200")
            ),
            memory_context_max_characters=int(
                source.get("MEMORY_CONTEXT_MAX_CHARACTERS", "4800")
            ),
            agent_mode=AgentMode(
                source.get("AGENT_MODE", "dynamic_planner").strip().lower()
            ),
            agent_max_handoffs=int(source.get("AGENT_MAX_HANDOFFS", "8")),
            planner_provider=PlannerProviderMode(
                source.get("PLANNER_PROVIDER", "mock").strip().lower()
            ),
            planner_model=source.get(
                "PLANNER_MODEL", "mock-dynamic-planner-v1"
            ).strip(),
            planner_timeout_seconds=float(
                source.get("PLANNER_TIMEOUT_SECONDS", "20")
            ),
            planner_max_attempts=int(source.get("PLANNER_MAX_ATTEMPTS", "2")),
            planner_max_output_tokens=int(
                source.get("PLANNER_MAX_OUTPUT_TOKENS", "1200")
            ),
            planner_max_decisions=int(
                source.get("PLANNER_MAX_DECISIONS", "12")
            ),
            planner_max_invalid_actions=int(
                source.get("PLANNER_MAX_INVALID_ACTIONS", "2")
            ),
            planner_max_repeated_actions=int(
                source.get("PLANNER_MAX_REPEATED_ACTIONS", "2")
            ),
            replanner_provider=ReplannerProviderMode(
                source.get("REPLANNER_PROVIDER", "deterministic").strip().lower()
            ),
            replanner_model=source.get(
                "REPLANNER_MODEL", "deterministic-replanner-v1"
            ).strip(),
            replanner_timeout_seconds=float(
                source.get("REPLANNER_TIMEOUT_SECONDS", "20")
            ),
            replanner_max_attempts=int(
                source.get("REPLANNER_MAX_ATTEMPTS", "2")
            ),
            replanner_max_output_tokens=int(
                source.get("REPLANNER_MAX_OUTPUT_TOKENS", "1600")
            ),
            max_evidence_records=int(source.get("MAX_EVIDENCE_RECORDS", "256")),
            max_observation_history=int(
                source.get("MAX_OBSERVATION_HISTORY", "8")
            ),
            max_planner_context_tokens=int(
                source.get("MAX_PLANNER_CONTEXT_TOKENS", "6000")
            ),
            max_replanner_context_tokens=int(
                source.get("MAX_REPLANNER_CONTEXT_TOKENS", "5000")
            ),
            dev_identity_enabled=_parse_bool(
                source.get("DEV_IDENTITY_ENABLED", "true"),
                name="DEV_IDENTITY_ENABLED",
            ),
            dev_tenant_id=source.get("DEV_TENANT_ID", "local").strip(),
            dev_user_id=source.get("DEV_USER_ID", "demo").strip(),
            mcp_allowed_hosts=_parse_csv(
                source.get(
                    "MCP_ALLOWED_HOSTS",
                    "127.0.0.1:*,localhost:*,[::1]:*,testserver,api:*",
                )
            ),
            mcp_allowed_origins=_parse_csv(
                source.get(
                    "MCP_ALLOWED_ORIGINS",
                    "http://127.0.0.1:*,http://localhost:*,http://[::1]:*",
                )
            ),
            async_execution_backend=AsyncExecutionBackend(
                source.get("ASYNC_EXECUTION_BACKEND", "local").strip().lower()
            ),
            redis_url=source.get(
                "REDIS_URL", "redis://localhost:6379/0"
            ).strip(),
            run_budget_profile=source.get(
                "RUN_BUDGET_PROFILE", "default-v1"
            ).strip(),
            run_max_graph_steps=int(source.get("RUN_MAX_GRAPH_STEPS", "120")),
            run_max_tool_calls=int(source.get("RUN_MAX_TOOL_CALLS", "160")),
            run_max_provider_attempts=int(
                source.get("RUN_MAX_PROVIDER_ATTEMPTS", "240")
            ),
            run_max_llm_calls=int(source.get("RUN_MAX_LLM_CALLS", "8")),
            run_max_llm_attempts=int(source.get("RUN_MAX_LLM_ATTEMPTS", "12")),
            run_max_llm_input_chars=int(
                source.get("RUN_MAX_LLM_INPUT_CHARS", "80000")
            ),
            run_max_input_tokens=int(source.get("RUN_MAX_INPUT_TOKENS", "40000")),
            run_max_output_tokens=int(
                source.get("RUN_MAX_OUTPUT_TOKENS", "12000")
            ),
            run_max_repair_rounds=int(
                source.get("RUN_MAX_REPAIR_ROUNDS", "4")
            ),
            run_max_interrupts=int(source.get("RUN_MAX_INTERRUPTS", "1")),
            run_max_checkpoint_writes=int(
                source.get("RUN_MAX_CHECKPOINT_WRITES", "160")
            ),
            run_max_trace_events=int(
                source.get("RUN_MAX_TRACE_EVENTS", "512")
            ),
            run_max_repeated_fingerprint_count=int(
                source.get("RUN_MAX_REPEATED_FINGERPRINT_COUNT", "2")
            ),
            run_deadline_seconds=float(
                source.get("RUN_DEADLINE_SECONDS", "120")
            ),
            trace_attribute_max_chars=int(
                source.get("TRACE_ATTRIBUTE_MAX_CHARS", "256")
            ),
            trace_file_enabled=_parse_bool(
                source.get("TRACE_FILE_ENABLED", "true"), name="TRACE_FILE_ENABLED"
            ),
            trace_file_dir=source.get("TRACE_FILE_DIR", ".data/traces").strip(),
        )
        settings.validate()
        return settings

    @property
    def effective_agent_mode(self) -> AgentMode:
        """把历史模式名映射到当前控制面，同时保留配置兼容性。"""

        return {
            AgentMode.SPECIALIST_SUBAGENTS: AgentMode.DYNAMIC_PLANNER,
            AgentMode.SHADOW_SUBAGENTS: AgentMode.SHADOW_DYNAMIC_PLANNER,
            AgentMode.SINGLE_GRAPH: AgentMode.FIXED_WORKFLOW,
        }.get(self.agent_mode, self.agent_mode)

    def validate(self) -> None:
        if self.provider is ProviderMode.AMAP and not self.amap_api_key:
            raise ValueError("AMAP_API_KEY is required when TRAVEL_PROVIDER=amap")
        if self.provider is ProviderMode.BAIDU and not self.baidu_api_key:
            raise ValueError("BAIDU_MAP_AK is required when TRAVEL_PROVIDER=baidu")
        if self.map_fallback_provider not in {"none", "baidu"}:
            raise ValueError("MAP_FALLBACK_PROVIDER must be none or baidu")
        if self.map_fallback_provider == "baidu" and not self.baidu_api_key:
            raise ValueError(
                "BAIDU_MAP_AK is required when MAP_FALLBACK_PROVIDER=baidu"
            )
        if self.weather_provider is WeatherProviderMode.AMAP and not self.amap_api_key:
            raise ValueError("AMAP_API_KEY is required when WEATHER_PROVIDER=amap")
        if self.weather_provider is WeatherProviderMode.QWEATHER and (
            not self.qweather_api_host or not self.qweather_token
        ):
            raise ValueError(
                "QWEATHER_API_HOST and QWEATHER_TOKEN are required when "
                "WEATHER_PROVIDER=qweather"
            )
        if self.weather_fallback_provider not in {"none", "qweather"}:
            raise ValueError(
                "WEATHER_FALLBACK_PROVIDER must be none or qweather"
            )
        if self.weather_fallback_provider == "qweather" and (
            not self.qweather_api_host or not self.qweather_token
        ):
            raise ValueError(
                "QWEATHER_API_HOST and QWEATHER_TOKEN are required when "
                "WEATHER_FALLBACK_PROVIDER=qweather"
            )
        if self.tool_timeout_seconds <= 0:
            raise ValueError("TOOL_TIMEOUT_SECONDS must be positive")
        if self.tool_max_attempts < 1:
            raise ValueError("TOOL_MAX_ATTEMPTS must be at least 1")
        if self.tool_backoff_base_seconds <= 0:
            raise ValueError("TOOL_BACKOFF_BASE_SECONDS must be positive")
        if self.tool_max_backoff_seconds <= 0:
            raise ValueError("TOOL_MAX_BACKOFF_SECONDS must be positive")
        if self.tool_max_concurrency <= 0:
            raise ValueError("TOOL_MAX_CONCURRENCY must be positive")
        if self.poi_cache_ttl_seconds <= 0:
            raise ValueError("POI_CACHE_TTL_SECONDS must be positive")
        if self.route_cache_ttl_seconds <= 0:
            raise ValueError("ROUTE_CACHE_TTL_SECONDS must be positive")
        if self.critic_provider is CriticProviderMode.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when CRITIC_PROVIDER=openai"
                )
            if not self.critic_model or self.critic_model == "mock-soft-critic-v1":
                raise ValueError(
                    "CRITIC_MODEL is required when CRITIC_PROVIDER=openai"
                )
        if self.critic_provider is CriticProviderMode.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when CRITIC_PROVIDER=deepseek"
                )
            if not self.critic_model or self.critic_model == "mock-soft-critic-v1":
                raise ValueError(
                    "CRITIC_MODEL is required when CRITIC_PROVIDER=deepseek"
                )
            if self.critic_model in {"deepseek-chat", "deepseek-reasoner"}:
                raise ValueError(
                    "CRITIC_MODEL uses a retired alias; choose an active model"
                )
            _validate_deepseek_base_url(self.deepseek_base_url)
        if self.critic_timeout_seconds <= 0:
            raise ValueError("CRITIC_TIMEOUT_SECONDS must be positive")
        if self.critic_max_attempts < 1:
            raise ValueError("CRITIC_MAX_ATTEMPTS must be at least 1")
        if self.critic_grounding_max_attempts not in {1, 2}:
            raise ValueError("CRITIC_GROUNDING_MAX_ATTEMPTS must be 1 or 2")
        if self.critic_max_input_chars < 1:
            raise ValueError("CRITIC_MAX_INPUT_CHARS must be positive")
        if self.critic_max_output_tokens < 1:
            raise ValueError("CRITIC_MAX_OUTPUT_TOKENS must be positive")
        if self.requirement_provider is RequirementProviderMode.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when REQUIREMENT_PROVIDER=openai"
                )
            if (
                not self.requirement_model
                or self.requirement_model == "mock-requirement-v1"
            ):
                raise ValueError(
                    "REQUIREMENT_MODEL is required when REQUIREMENT_PROVIDER=openai"
                )
        if self.requirement_provider is RequirementProviderMode.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError(
                    "DEEPSEEK_API_KEY is required when "
                    "REQUIREMENT_PROVIDER=deepseek"
                )
            if not self.deepseek_model:
                raise ValueError(
                    "DEEPSEEK_MODEL is required when "
                    "REQUIREMENT_PROVIDER=deepseek"
                )
            if self.deepseek_model in {"deepseek-chat", "deepseek-reasoner"}:
                raise ValueError(
                    "DEEPSEEK_MODEL uses a retired alias; choose an active model"
                )
            _validate_deepseek_base_url(self.deepseek_base_url)
        if self.requirement_timeout_seconds <= 0:
            raise ValueError("REQUIREMENT_TIMEOUT_SECONDS must be positive")
        if self.requirement_max_attempts < 1:
            raise ValueError("REQUIREMENT_MAX_ATTEMPTS must be at least 1")
        if self.requirement_backoff_base_seconds < 0:
            raise ValueError("REQUIREMENT_BACKOFF_BASE_SECONDS must be non-negative")
        if self.requirement_max_backoff_seconds < 0:
            raise ValueError("REQUIREMENT_MAX_BACKOFF_SECONDS must be non-negative")
        if self.edit_provider is EditProviderMode.OPENAI:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when EDIT_PROVIDER=openai")
            if not self.edit_model or self.edit_model == "mock-plan-edit-v1":
                raise ValueError("EDIT_MODEL is required when EDIT_PROVIDER=openai")
        if self.edit_provider is EditProviderMode.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when EDIT_PROVIDER=deepseek")
            if not self.edit_model or self.edit_model == "mock-plan-edit-v1":
                raise ValueError("EDIT_MODEL is required when EDIT_PROVIDER=deepseek")
            if self.edit_model in {"deepseek-chat", "deepseek-reasoner"}:
                raise ValueError("EDIT_MODEL uses a retired alias; choose an active model")
            _validate_deepseek_base_url(self.deepseek_base_url)
        if self.edit_timeout_seconds <= 0:
            raise ValueError("EDIT_TIMEOUT_SECONDS must be positive")
        if self.edit_max_attempts < 1:
            raise ValueError("EDIT_MAX_ATTEMPTS must be at least 1")
        if self.edit_max_output_tokens < 1:
            raise ValueError("EDIT_MAX_OUTPUT_TOKENS must be positive")
        if not 1 <= self.plan_max_versions <= 100:
            raise ValueError("PLAN_MAX_VERSIONS must be between 1 and 100")
        if not 1 <= self.plan_max_affected_days <= 7:
            raise ValueError("PLAN_MAX_AFFECTED_DAYS must be between 1 and 7")
        if self.weather_cache_ttl_seconds < 1:
            raise ValueError("WEATHER_CACHE_TTL_SECONDS must be positive")
        if self.weather_stale_max_seconds <= self.weather_cache_ttl_seconds:
            raise ValueError(
                "WEATHER_STALE_MAX_SECONDS must exceed WEATHER_CACHE_TTL_SECONDS"
            )
        if not 1 <= self.weather_max_events <= 500:
            raise ValueError("WEATHER_MAX_EVENTS must be between 1 and 500")
        if not 1 <= self.weather_max_poi_searches <= 10:
            raise ValueError("WEATHER_MAX_POI_SEARCHES must be between 1 and 10")
        if not 1 <= self.weather_max_alternatives <= 20:
            raise ValueError("WEATHER_MAX_ALTERNATIVES must be between 1 and 20")
        if not 0 < self.weather_exposure_min_confidence <= 1:
            raise ValueError(
                "WEATHER_EXPOSURE_MIN_CONFIDENCE must be in the interval (0, 1]"
            )
        if (
            self.checkpoint_backend is CheckpointBackend.SQLITE
            and not self.checkpoint_sqlite_path
        ):
            raise ValueError(
                "CHECKPOINT_SQLITE_PATH is required when CHECKPOINT_BACKEND=sqlite"
            )
        if self.plan_store_backend is RunStoreBackend.SQLITE and not self.plan_sqlite_path:
            raise ValueError("PLAN_SQLITE_PATH is required when PLAN_STORE_BACKEND=sqlite")
        if self.run_store_backend is RunStoreBackend.SQLITE and not self.run_sqlite_path:
            raise ValueError("RUN_SQLITE_PATH is required when RUN_STORE_BACKEND=sqlite")
        if (
            self.memory_store_backend is RunStoreBackend.SQLITE
            and not self.memory_sqlite_path
        ):
            raise ValueError(
                "MEMORY_SQLITE_PATH is required when MEMORY_STORE_BACKEND=sqlite"
            )
        if (
            RunStoreBackend.POSTGRES
            in {
                self.plan_store_backend,
                self.run_store_backend,
                self.memory_store_backend,
            }
            and not self.database_url
        ):
            raise ValueError("DATABASE_URL is required for PostgreSQL repositories")
        if not 32 <= self.memory_context_max_tokens <= 100_000:
            raise ValueError(
                "MEMORY_CONTEXT_MAX_TOKENS must be between 32 and 100000"
            )
        if not 128 <= self.memory_context_max_characters <= 400_000:
            raise ValueError(
                "MEMORY_CONTEXT_MAX_CHARACTERS must be between 128 and 400000"
            )
        if not 1 <= self.agent_max_handoffs <= 100:
            raise ValueError("AGENT_MAX_HANDOFFS must be between 1 and 100")
        if not self.planner_model:
            raise ValueError("PLANNER_MODEL must not be blank")
        if self.planner_provider is PlannerProviderMode.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when PLANNER_PROVIDER=openai")
        if self.planner_provider is PlannerProviderMode.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when PLANNER_PROVIDER=deepseek")
            _validate_deepseek_base_url(self.deepseek_base_url)
        if self.planner_timeout_seconds <= 0:
            raise ValueError("PLANNER_TIMEOUT_SECONDS must be positive")
        if self.planner_max_attempts < 1:
            raise ValueError("PLANNER_MAX_ATTEMPTS must be at least 1")
        if self.planner_max_output_tokens < 1:
            raise ValueError("PLANNER_MAX_OUTPUT_TOKENS must be positive")
        if not 1 <= self.planner_max_decisions <= 100:
            raise ValueError("PLANNER_MAX_DECISIONS must be between 1 and 100")
        if not 0 <= self.planner_max_invalid_actions <= 20:
            raise ValueError("PLANNER_MAX_INVALID_ACTIONS must be between 0 and 20")
        if not 1 <= self.planner_max_repeated_actions <= 20:
            raise ValueError("PLANNER_MAX_REPEATED_ACTIONS must be between 1 and 20")
        if not self.replanner_model:
            raise ValueError("REPLANNER_MODEL must not be blank")
        if self.replanner_provider is ReplannerProviderMode.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when REPLANNER_PROVIDER=openai")
        if self.replanner_provider is ReplannerProviderMode.DEEPSEEK:
            if not self.deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is required when REPLANNER_PROVIDER=deepseek")
            _validate_deepseek_base_url(self.deepseek_base_url)
        if self.replanner_timeout_seconds <= 0:
            raise ValueError("REPLANNER_TIMEOUT_SECONDS must be positive")
        if self.replanner_max_attempts < 1:
            raise ValueError("REPLANNER_MAX_ATTEMPTS must be at least 1")
        if self.replanner_max_output_tokens < 1:
            raise ValueError("REPLANNER_MAX_OUTPUT_TOKENS must be positive")
        if not 1 <= self.max_evidence_records <= 10_000:
            raise ValueError("MAX_EVIDENCE_RECORDS must be between 1 and 10000")
        if not 1 <= self.max_observation_history <= 100:
            raise ValueError("MAX_OBSERVATION_HISTORY must be between 1 and 100")
        if self.dev_identity_enabled and (
            not self.dev_tenant_id or not self.dev_user_id
        ):
            raise ValueError(
                "DEV_TENANT_ID and DEV_USER_ID are required when dev identity is enabled"
            )
        if not self.mcp_allowed_hosts:
            raise ValueError("MCP_ALLOWED_HOSTS must contain at least one host")
        if self.async_execution_backend is AsyncExecutionBackend.REDIS and not self.redis_url:
            raise ValueError("REDIS_URL is required when ASYNC_EXECUTION_BACKEND=redis")
        if not self.run_budget_profile:
            raise ValueError("RUN_BUDGET_PROFILE must not be blank")
        if not 32 <= self.trace_attribute_max_chars <= 2048:
            raise ValueError("TRACE_ATTRIBUTE_MAX_CHARS must be between 32 and 2048")
        if self.trace_file_enabled and not self.trace_file_dir:
            raise ValueError("TRACE_FILE_DIR must not be blank when file tracing is enabled")
        self.execution_budget()
        PlanningPolicy(
            poi_query_limit=self.poi_query_limit,
            poi_candidate_limit=self.poi_candidate_limit,
            route_strategy=self.amap_driving_strategy,
            max_walking_leg_meters=self.max_walking_leg_meters,
            use_real_walking_routes=self.use_real_walking_routes,
            poi_max_queries=self.poi_max_queries,
        )
        OptimizationBudget(
            max_solve_ms=self.optimization_max_solve_ms,
            max_search_states=self.optimization_max_search_states,
            candidate_limit=self.optimization_candidate_limit,
            variant_count=self.optimization_variant_count,
        )
        from travel_agent.critique.quality import CriticPolicy

        CriticPolicy(
            quality_threshold=self.critic_quality_threshold,
            min_improvement=self.critic_min_improvement,
            max_soft_replan_rounds=self.max_soft_replan_rounds,
            grounding_max_attempts=self.critic_grounding_max_attempts,
        )

    def execution_budget(self) -> ExecutionBudget:
        return ExecutionBudget(
            profile=self.run_budget_profile,
            max_graph_steps=self.run_max_graph_steps,
            max_tool_calls=self.run_max_tool_calls,
            max_provider_attempts=self.run_max_provider_attempts,
            max_llm_calls=self.run_max_llm_calls,
            max_llm_attempts=self.run_max_llm_attempts,
            max_llm_input_chars=self.run_max_llm_input_chars,
            max_input_tokens=self.run_max_input_tokens,
            max_output_tokens=self.run_max_output_tokens,
            max_repair_rounds=self.run_max_repair_rounds,
            max_interrupts=self.run_max_interrupts,
            max_checkpoint_writes=self.run_max_checkpoint_writes,
            max_trace_events=self.run_max_trace_events,
            max_repeated_fingerprint_count=(
                self.run_max_repeated_fingerprint_count
            ),
            max_agent_decisions=self.planner_max_decisions,
            max_invalid_actions=self.planner_max_invalid_actions,
            max_evidence_records=self.max_evidence_records,
            max_observation_history=self.max_observation_history,
            max_planner_context_tokens=self.max_planner_context_tokens,
            max_replanner_context_tokens=self.max_replanner_context_tokens,
            deadline_ms=round(self.run_deadline_seconds * 1000),
        )


def _validate_deepseek_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "DEEPSEEK_BASE_URL must be an HTTPS URL without credentials, "
            "query, or fragment"
        )


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())

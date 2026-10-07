from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from langgraph.graph.state import CompiledStateGraph

from travel_agent.agents.orchestrator import SpecialistExecutor
from travel_agent.config import (
    AgentMode,
    CriticProviderMode,
    EditProviderMode,
    PlannerProviderMode,
    ReplannerProviderMode,
    Settings,
    WeatherProviderMode,
)
from travel_agent.agents.planner.gateway import PlannerGateway
from travel_agent.agents.planner.providers.mock import MockPlannerModel
from travel_agent.agents.planner.providers.openai import OpenAIPlannerModel
from travel_agent.agents.planner.providers.deepseek import DeepSeekPlannerModel
from travel_agent.agents.replanner.gateway import ReplannerGateway
from travel_agent.agents.replanner.providers.mock import MockReplannerModel
from travel_agent.agents.replanner.providers.openai import OpenAIReplannerModel
from travel_agent.agents.replanner.providers.deepseek import DeepSeekReplannerModel
from travel_agent.evidence.repository import InMemoryEvidenceRepository
from travel_agent.evidence.policy import derive_evidence_gaps
from travel_agent.agents.actions import AgentPhase
from travel_agent.agents.context import DynamicPlannerContext, PlannerContextManifest
from travel_agent.execution.context import current_run_context
from travel_agent.execution.models import ExecutionUsage
from travel_agent.memory.models import AgentRole
from travel_agent.graph.agentic_workflow import build_agentic_workflow, run_agentic_planning
from travel_agent.tools.agent_executor import AgentToolExecutor
from travel_agent.tools.registry import default_agent_tool_registry
from travel_agent.tools.registry import ToolRegistry
from travel_agent.critique.evidence import EvidenceBudget
from travel_agent.critique.gateway import CriticGateway
from travel_agent.critique.protocols import CriticModel
from travel_agent.critique.providers.deepseek import DeepSeekCriticModel
from travel_agent.critique.providers.mock import MockCriticModel
from travel_agent.critique.providers.openai import OpenAICriticModel
from travel_agent.critique.quality import CriticPolicy
from travel_agent.domain.models import PlanningRequest, PlanningResponse
from travel_agent.domain.lifecycle_models import (
    LifecycleResumeRequest,
    PlanSessionResponse,
)
from travel_agent.domain.weather_models import (
    WeatherEventView,
    WeatherRefreshRequest,
    WeatherStateView,
)
from travel_agent.domain.optimization_models import OptimizationBudget
from travel_agent.domain.tool_models import ProviderMode
from travel_agent.graph.workflow import build_workflow, run_planning
from travel_agent.graph.evaluation import PlanningEvaluationOverrides
from travel_agent.execution.coordinator import ExecutionResult, RunCoordinator
from travel_agent.execution.faults import FaultPlan
from travel_agent.execution.models import (
    AgentRunRecord,
    ExecutionBudget,
    RunKind,
    TraceEvent,
)
from travel_agent.execution.repository import RunRepository, open_run_repository
from travel_agent.edits.gateway import EditGateway
from travel_agent.edits.protocols import EditModel
from travel_agent.edits.providers.mock import MockEditModel
from travel_agent.edits.providers.openai import OpenAIEditModel
from travel_agent.edits.providers.deepseek import DeepSeekEditModel
from travel_agent.lifecycle.repository import PlanRepository, open_plan_repository
from travel_agent.lifecycle.service import PlanLifecycleService
from travel_agent.lifecycle.workflow import build_lifecycle_workflow
from travel_agent.planning.defaults import POIDefaultPolicy
from travel_agent.planning.kernel import PlanningKernelService
from travel_agent.planning.quality import QualityReviewService
from travel_agent.planning.policy import PlanningPolicy
from travel_agent.tools.gateway import ToolGateway, build_gateway
from travel_agent.tools.protocols import POIProvider, RouteProvider
from travel_agent.tools.providers.amap import (
    AMapClient,
    AMapPOIProvider,
    AMapRouteProvider,
)
from travel_agent.tools.providers.mock import MockPOIProvider, MockRouteProvider
from travel_agent.tools.providers.qunar import QunarScenicPOIProvider
from travel_agent.requirements.gateway import RequirementGateway
from travel_agent.requirements.checkpoints import open_requirement_checkpointer
from travel_agent.requirements.models import (
    ClarificationResumeRequest,
    NaturalPlanningRequest,
    NaturalPlanningResponse,
    RequirementProviderMode,
)
from travel_agent.tools.providers.baidu import (
    BaiduMapClient,
    BaiduPOIProvider,
    BaiduRouteProvider,
)
from travel_agent.tools.providers.chain import (
    POIProviderChain,
    RouteProviderChain,
    WeatherProviderChain,
)
from travel_agent.requirements.protocols import RequirementModel
from travel_agent.requirements.providers.mock import MockRequirementModel
from travel_agent.requirements.providers.openai import OpenAIRequirementModel
from travel_agent.requirements.providers.deepseek import DeepSeekRequirementModel
from travel_agent.requirements.workflow import (
    build_requirement_workflow,
    resume_natural_planning,
    run_natural_planning,
)
from travel_agent.weather.gateway import WeatherToolGateway, build_weather_gateway
from travel_agent.weather.protocols import WeatherProvider
from travel_agent.weather.providers.amap import AMapWeatherProvider
from travel_agent.weather.providers.mock import MockWeatherProvider
from travel_agent.weather.providers.qweather import QWeatherProvider
from travel_agent.identity.models import Principal
from travel_agent.memory.repository import (
    PreferenceRepository,
    open_preference_repository,
)
from travel_agent.memory.service import PreferenceMemoryService


async def _run_shadow_decision(
    planner_gateway: PlannerGateway,
    registry: ToolRegistry,
    request: PlanningRequest,
    *,
    thread_id: str,
) -> str:
    """只记录影子动作，不分发 Tool，也不让影子结果进入计划状态。"""
    gaps = derive_evidence_gaps(request.trip, ())
    tools = registry.manifest(role=AgentRole.PLANNER, phase=AgentPhase.RESEARCHING)
    run = current_run_context()
    usage = run.ledger.snapshot() if run is not None else ExecutionUsage()
    context = DynamicPlannerContext(
        goal=request.trip,
        phase=AgentPhase.RESEARCHING,
        evidence_gaps=gaps,
        evidence_catalog=(),
        recent_observations=(),
        tool_manifest=tools,
        budget_remaining=usage,
        context_manifest=PlannerContextManifest(
            tool_names=tuple(item.name for item in tools),
        ),
    )
    decision = await planner_gateway.decide(
        context, thread_id=f"{thread_id}:shadow", decision_round=1
    )
    return decision.action.kind


@dataclass(slots=True)
class PlanningRuntime:
    """持有一套显式选定、由应用生命周期共享的规划依赖。"""

    poi_provider: POIProvider
    route_provider: RouteProvider
    gateway: ToolGateway
    workflow: CompiledStateGraph
    client: httpx.AsyncClient | None
    agent_mode: AgentMode = AgentMode.DYNAMIC_PLANNER
    agentic_workflow: CompiledStateGraph | None = None
    planner_gateway: PlannerGateway | None = None
    tool_registry: ToolRegistry | None = None
    auxiliary_client: httpx.AsyncClient | None = None
    weather_provider: WeatherProvider | None = None
    weather_gateway: WeatherToolGateway | None = None
    requirement_model: RequirementModel | None = None
    requirement_gateway: RequirementGateway | None = None
    requirement_workflow: CompiledStateGraph | None = None
    model_client: Any | None = None
    planner_model_client: Any | None = None
    replanner_model_client: Any | None = None
    replanner_gateway: ReplannerGateway | None = None
    critic_model: CriticModel | None = None
    critic_gateway: CriticGateway | None = None
    critic_model_client: Any | None = None
    edit_model: EditModel | None = None
    edit_gateway: EditGateway | None = None
    edit_model_client: Any | None = None
    plan_repository: PlanRepository | None = None
    lifecycle_workflow: CompiledStateGraph | None = None
    lifecycle_service: PlanLifecycleService | None = None
    checkpoint_context: Any | None = field(default=None, repr=False)
    repository_context: Any | None = field(default=None, repr=False)
    run_repository: RunRepository | None = None
    run_coordinator: RunCoordinator | None = None
    run_repository_context: Any | None = field(default=None, repr=False)
    preference_repository: PreferenceRepository | None = None
    preference_service: PreferenceMemoryService | None = None
    preference_repository_context: Any | None = field(default=None, repr=False)
    specialist_executor: SpecialistExecutor | None = None
    resume_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        evaluation_overrides: PlanningEvaluationOverrides | None = None,
        tool_cache_enabled: bool = True,
    ) -> "PlanningRuntime":
        settings.validate()
        agent_mode = settings.effective_agent_mode
        client: httpx.AsyncClient | None = None
        model_client: Any | None = None
        planner_model_client: Any | None = None
        replanner_model_client: Any | None = None
        critic_model_client: Any | None = None
        edit_model_client: Any | None = None
        auxiliary_client: httpx.AsyncClient | None = None
        checkpoint_context: Any | None = None
        repository_context: Any | None = None
        run_repository_context: Any | None = None
        preference_repository_context: Any | None = None
        checkpoint_entered = False
        repository_entered = False
        run_repository_entered = False
        preference_repository_entered = False
        try:
            amap_client: AMapClient | None = None
            if (
                settings.provider is ProviderMode.AMAP
                or settings.weather_provider is WeatherProviderMode.AMAP
            ):
                assert settings.amap_api_key is not None
                client = httpx.AsyncClient(base_url="https://restapi.amap.com")
                amap_client = AMapClient(
                    client,
                    api_key=settings.amap_api_key,
                    timeout_seconds=settings.tool_timeout_seconds,
                )
            baidu_client: BaiduMapClient | None = None
            if (
                settings.provider is ProviderMode.BAIDU
                or settings.map_fallback_provider == "baidu"
            ):
                auxiliary_client = httpx.AsyncClient(
                    timeout=settings.tool_timeout_seconds
                )
                assert settings.baidu_api_key is not None
                baidu_client = BaiduMapClient(
                    auxiliary_client, settings.baidu_api_key
                )
            if settings.provider is ProviderMode.MOCK:
                poi_provider: POIProvider = MockPOIProvider()
                route_provider: RouteProvider = MockRouteProvider()
            elif settings.provider is ProviderMode.AMAP:
                assert amap_client is not None
                assert client is not None
                poi_provider = QunarScenicPOIProvider(AMapPOIProvider(amap_client), client)
                route_provider = AMapRouteProvider(amap_client)
            else:
                assert baidu_client is not None
                poi_provider = BaiduPOIProvider(baidu_client)
                route_provider = BaiduRouteProvider(baidu_client)
            if settings.map_fallback_provider == "baidu":
                assert baidu_client is not None
                poi_provider = POIProviderChain(
                    (poi_provider, BaiduPOIProvider(baidu_client))
                )
                route_provider = RouteProviderChain(
                    (route_provider, BaiduRouteProvider(baidu_client))
                )

            gateway = build_gateway(
                settings,
                poi_provider,
                route_provider,
                cache_enabled=tool_cache_enabled,
            )
            if settings.weather_provider is WeatherProviderMode.MOCK:
                weather_provider: WeatherProvider = MockWeatherProvider()
            elif settings.weather_provider is WeatherProviderMode.AMAP:
                assert amap_client is not None
                weather_provider = AMapWeatherProvider(amap_client)
            else:
                if auxiliary_client is None:
                    auxiliary_client = httpx.AsyncClient(
                        timeout=settings.tool_timeout_seconds
                    )
                assert settings.qweather_token is not None
                weather_provider = QWeatherProvider(
                    auxiliary_client,
                    api_host=settings.qweather_api_host,
                    token=settings.qweather_token,
                )
            if settings.weather_fallback_provider == "qweather":
                if auxiliary_client is None:
                    auxiliary_client = httpx.AsyncClient(
                        timeout=settings.tool_timeout_seconds
                    )
                assert settings.qweather_token is not None
                weather_provider = WeatherProviderChain(
                    (
                        weather_provider,
                        QWeatherProvider(
                            auxiliary_client,
                            api_host=settings.qweather_api_host,
                            token=settings.qweather_token,
                        ),
                    )
                )
            weather_gateway = build_weather_gateway(settings, weather_provider)
            defaults = POIDefaultPolicy(settings.unknown_fact_policy)
            policy = PlanningPolicy(
                poi_query_limit=settings.poi_query_limit,
                poi_candidate_limit=settings.poi_candidate_limit,
                route_strategy=settings.amap_driving_strategy,
                max_walking_leg_meters=settings.max_walking_leg_meters,
                use_real_walking_routes=settings.use_real_walking_routes,
                poi_max_queries=settings.poi_max_queries,
            )
            optimization_budget = OptimizationBudget(
                max_solve_ms=settings.optimization_max_solve_ms,
                max_search_states=settings.optimization_max_search_states,
                candidate_limit=settings.optimization_candidate_limit,
                variant_count=settings.optimization_variant_count,
            )
            critic_model: CriticModel | None
            critic_gateway: CriticGateway | None
            if settings.critic_provider is CriticProviderMode.DISABLED:
                critic_model = None
                critic_gateway = None
            elif settings.critic_provider is CriticProviderMode.MOCK:
                critic_model = MockCriticModel()
                critic_gateway = CriticGateway(
                    model=critic_model,
                    timeout_seconds=settings.critic_timeout_seconds,
                    max_attempts=settings.critic_max_attempts,
                )
            else:
                try:
                    from openai import AsyncOpenAI
                except ImportError as error:
                    extra = (
                        "llm-openai"
                        if settings.critic_provider is CriticProviderMode.OPENAI
                        else "llm-deepseek"
                    )
                    raise RuntimeError(
                        f"Install the {extra} extra to use "
                        f"CRITIC_PROVIDER={settings.critic_provider.value}"
                    ) from error
                if settings.critic_provider is CriticProviderMode.OPENAI:
                    assert settings.openai_api_key is not None
                    critic_model_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        timeout=settings.critic_timeout_seconds,
                        max_retries=0,
                    )
                    critic_model = OpenAICriticModel(
                        client=critic_model_client,
                        model=settings.critic_model,
                    )
                else:
                    assert settings.deepseek_api_key is not None
                    critic_model_client = AsyncOpenAI(
                        api_key=settings.deepseek_api_key,
                        base_url=settings.deepseek_base_url,
                        timeout=settings.critic_timeout_seconds,
                        max_retries=0,
                    )
                    critic_model = DeepSeekCriticModel(
                        client=critic_model_client,
                        model=settings.critic_model,
                        max_tokens=settings.critic_max_output_tokens,
                    )
                critic_gateway = CriticGateway(
                    model=critic_model,
                    timeout_seconds=settings.critic_timeout_seconds,
                    max_attempts=settings.critic_max_attempts,
                )
            critic_policy = CriticPolicy(
                quality_threshold=settings.critic_quality_threshold,
                min_improvement=settings.critic_min_improvement,
                max_soft_replan_rounds=settings.max_soft_replan_rounds,
                grounding_max_attempts=settings.critic_grounding_max_attempts,
            )
            specialist_executor = None
            workflow = build_workflow(
                gateway,
                defaults,
                policy,
                optimization_budget=optimization_budget,
                critic_gateway=critic_gateway,
                critic_policy=critic_policy,
                evidence_budget=EvidenceBudget(
                    max_input_chars=settings.critic_max_input_chars
                ),
                evaluation_overrides=evaluation_overrides,
                agent_mode=agent_mode,
                specialist_executor=specialist_executor,
            )
            if settings.requirement_provider is RequirementProviderMode.MOCK:
                requirement_model: RequirementModel = MockRequirementModel()
            else:
                try:
                    from openai import AsyncOpenAI
                except ImportError as error:
                    extra = (
                        "llm-openai"
                        if settings.requirement_provider
                        is RequirementProviderMode.OPENAI
                        else "llm-deepseek"
                    )
                    raise RuntimeError(
                        f"Install the {extra} extra to use "
                        f"REQUIREMENT_PROVIDER={settings.requirement_provider.value}"
                    ) from error
                if settings.requirement_provider is RequirementProviderMode.OPENAI:
                    assert settings.openai_api_key is not None
                    model_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        timeout=settings.requirement_timeout_seconds,
                        max_retries=0,
                    )
                    requirement_model = OpenAIRequirementModel(
                        client=model_client,
                        model=settings.requirement_model,
                    )
                else:
                    assert settings.deepseek_api_key is not None
                    model_client = AsyncOpenAI(
                        api_key=settings.deepseek_api_key,
                        base_url=settings.deepseek_base_url,
                        timeout=settings.requirement_timeout_seconds,
                        max_retries=0,
                    )
                    requirement_model = DeepSeekRequirementModel(
                        client=model_client,
                        model=settings.deepseek_model,
                    )
            requirement_gateway = RequirementGateway(
                model=requirement_model,
                timeout_seconds=settings.requirement_timeout_seconds,
                max_attempts=settings.requirement_max_attempts,
                base_delay_seconds=settings.requirement_backoff_base_seconds,
                max_delay_seconds=settings.requirement_max_backoff_seconds,
            )
            preference_repository_context = open_preference_repository(
                backend=settings.memory_store_backend.value,
                sqlite_path=settings.memory_sqlite_path,
                database_url=settings.database_url,
            )
            preference_repository = (
                await preference_repository_context.__aenter__()
            )
            preference_repository_entered = True
            preference_service = PreferenceMemoryService(
                preference_repository,
                context_max_tokens=settings.memory_context_max_tokens,
                context_max_characters=settings.memory_context_max_characters,
            )
            agentic_workflow: CompiledStateGraph | None = None
            planner_gateway: PlannerGateway | None = None
            tool_registry: ToolRegistry | None = None
            if agent_mode in {
                AgentMode.DYNAMIC_PLANNER,
                AgentMode.SHADOW_DYNAMIC_PLANNER,
            }:
                tool_registry = default_agent_tool_registry()
                if settings.planner_provider is PlannerProviderMode.MOCK:
                    planner_model = MockPlannerModel()
                else:
                    try:
                        from openai import AsyncOpenAI
                    except ImportError as error:
                        raise RuntimeError(
                            "Install the llm-openai extra to use a remote planner"
                        ) from error
                    if settings.planner_provider is PlannerProviderMode.OPENAI:
                        assert settings.openai_api_key is not None
                        planner_model_client = AsyncOpenAI(
                            api_key=settings.openai_api_key,
                            timeout=settings.planner_timeout_seconds,
                            max_retries=0,
                        )
                        planner_model = OpenAIPlannerModel(
                            client=planner_model_client,
                            model=settings.planner_model,
                        )
                    else:
                        assert settings.deepseek_api_key is not None
                        planner_model_client = AsyncOpenAI(
                            api_key=settings.deepseek_api_key,
                            base_url=settings.deepseek_base_url,
                            timeout=settings.planner_timeout_seconds,
                            max_retries=0,
                        )
                        planner_model = DeepSeekPlannerModel(
                            client=planner_model_client,
                            model=settings.planner_model,
                            max_tokens=settings.planner_max_output_tokens,
                        )
                planner_gateway = PlannerGateway(
                    model=planner_model,
                    timeout_seconds=settings.planner_timeout_seconds,
                    max_attempts=settings.planner_max_attempts,
                )
                replanner_gateway: ReplannerGateway | None = None
                if settings.replanner_provider is ReplannerProviderMode.MOCK:
                    replanner_model = MockReplannerModel()
                elif settings.replanner_provider in {
                    ReplannerProviderMode.OPENAI,
                    ReplannerProviderMode.DEEPSEEK,
                }:
                    from openai import AsyncOpenAI
                    if settings.replanner_provider is ReplannerProviderMode.OPENAI:
                        assert settings.openai_api_key is not None
                        replanner_model_client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.replanner_timeout_seconds, max_retries=0)
                        replanner_model = OpenAIReplannerModel(client=replanner_model_client, model=settings.replanner_model)
                    else:
                        assert settings.deepseek_api_key is not None
                        replanner_model_client = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url, timeout=settings.replanner_timeout_seconds, max_retries=0)
                        replanner_model = DeepSeekReplannerModel(client=replanner_model_client, model=settings.replanner_model, max_tokens=settings.replanner_max_output_tokens)
                else:
                    replanner_model = None
                if replanner_model is not None:
                    replanner_gateway = ReplannerGateway(
                        model=replanner_model,
                        timeout_seconds=settings.replanner_timeout_seconds,
                        max_attempts=settings.replanner_max_attempts,
                    )
                planning_kernel = PlanningKernelService(
                    gateway=gateway,
                    defaults=defaults,
                    policy=policy,
                    optimization_budget=optimization_budget,
                )
                agentic_workflow = build_agentic_workflow(
                    planning_kernel=planning_kernel,
                    planner_gateway=planner_gateway,
                    tool_executor=AgentToolExecutor(
                        registry=tool_registry,
                        gateway=gateway,
                        defaults=defaults,
                        planning_policy=policy,
                        optimization_budget=optimization_budget,
                        weather_gateway=weather_gateway,
                        preference_service=preference_service,
                    ),
                    evidence_repository=InMemoryEvidenceRepository(
                        max_records_per_run=settings.max_evidence_records
                    ),
                    max_observation_history=settings.max_observation_history,
                    max_invalid_actions=settings.planner_max_invalid_actions,
                    replanner_gateway=replanner_gateway,
                    quality_service=QualityReviewService(
                        critic_gateway=critic_gateway,
                        policy=critic_policy,
                        evidence_budget=EvidenceBudget(
                            max_input_chars=settings.critic_max_input_chars
                        ),
                    ),
                    max_planner_context_tokens=settings.max_planner_context_tokens,
                )
            else:
                replanner_gateway = None

            async def selected_planning_runner(
                request: PlanningRequest, thread_id: str
            ) -> PlanningResponse:
                if (
                    agent_mode is AgentMode.DYNAMIC_PLANNER
                    and agentic_workflow is not None
                ):
                    return await run_agentic_planning(
                        agentic_workflow, request, thread_id=thread_id
                    )
                if agent_mode is AgentMode.SHADOW_DYNAMIC_PLANNER:
                    assert planner_gateway is not None and tool_registry is not None
                    shadow_action = await _run_shadow_decision(
                        planner_gateway, tool_registry, request, thread_id=thread_id
                    )
                    response = await run_planning(workflow, request, thread_id=thread_id)
                    return response.model_copy(update={
                        "agent_mode": AgentMode.SHADOW_DYNAMIC_PLANNER.value,
                        "decision_count": 1,
                        "action_summary": (shadow_action,),
                    })
                return await run_planning(workflow, request, thread_id=thread_id)

            checkpoint_context = open_requirement_checkpointer(settings)
            requirement_checkpointer = await checkpoint_context.__aenter__()
            checkpoint_entered = True
            requirement_workflow = build_requirement_workflow(
                requirement_gateway=requirement_gateway,
                tool_gateway=gateway,
                planning_workflow=workflow,
                planning_runner=selected_planning_runner,
                memory_service=preference_service,
                checkpointer=requirement_checkpointer,
            )
            if settings.edit_provider is EditProviderMode.MOCK:
                edit_model: EditModel = MockEditModel()
            else:
                try:
                    from openai import AsyncOpenAI
                except ImportError as error:
                    extra = (
                        "llm-openai"
                        if settings.edit_provider is EditProviderMode.OPENAI
                        else "llm-deepseek"
                    )
                    raise RuntimeError(
                        f"Install the {extra} extra to use "
                        f"EDIT_PROVIDER={settings.edit_provider.value}"
                    ) from error
                if settings.edit_provider is EditProviderMode.OPENAI:
                    assert settings.openai_api_key is not None
                    edit_model_client = AsyncOpenAI(
                        api_key=settings.openai_api_key,
                        timeout=settings.edit_timeout_seconds,
                        max_retries=0,
                    )
                    edit_model = OpenAIEditModel(
                        client=edit_model_client, model=settings.edit_model
                    )
                else:
                    assert settings.deepseek_api_key is not None
                    edit_model_client = AsyncOpenAI(
                        api_key=settings.deepseek_api_key,
                        base_url=settings.deepseek_base_url,
                        timeout=settings.edit_timeout_seconds,
                        max_retries=0,
                    )
                    edit_model = DeepSeekEditModel(
                        client=edit_model_client,
                        model=settings.edit_model,
                        max_tokens=settings.edit_max_output_tokens,
                    )
            edit_gateway = EditGateway(
                model=edit_model,
                timeout_seconds=settings.edit_timeout_seconds,
                max_attempts=settings.edit_max_attempts,
            )
            plan_store_backend = (
                settings.checkpoint_backend.value
                if settings.plan_store_backend.value == "memory"
                and settings.checkpoint_backend.value == "sqlite"
                else settings.plan_store_backend.value
            )
            repository_context = open_plan_repository(
                backend=plan_store_backend,
                sqlite_path=settings.plan_sqlite_path,
                database_url=settings.database_url,
            )
            plan_repository = await repository_context.__aenter__()
            repository_entered = True
            lifecycle_workflow = build_lifecycle_workflow(
                repository=plan_repository,
                tool_gateway=gateway,
                edit_gateway=edit_gateway,
                planning_policy=policy,
                poi_default_policy=defaults,
                critic_gateway=critic_gateway,
                critic_policy=critic_policy,
                evidence_budget=EvidenceBudget(
                    max_candidates=1,
                    max_input_chars=settings.critic_max_input_chars,
                ),
                max_affected_days=settings.plan_max_affected_days,
                max_versions=settings.plan_max_versions,
                weather_gateway=weather_gateway,
                weather_max_events=settings.weather_max_events,
                weather_max_poi_searches=settings.weather_max_poi_searches,
                weather_max_alternatives=settings.weather_max_alternatives,
                weather_exposure_min_confidence=(
                    settings.weather_exposure_min_confidence
                ),
                checkpointer=requirement_checkpointer,
            )
            lifecycle_service = PlanLifecycleService(
                repository=plan_repository,
                planning_workflow=(
                    agentic_workflow
                    if agent_mode is AgentMode.DYNAMIC_PLANNER
                    and agentic_workflow is not None
                    else workflow
                ),
                planning_runner=selected_planning_runner,
                lifecycle_workflow=lifecycle_workflow,
                requirement_workflow=requirement_workflow,
                weather_stale_max_seconds=settings.weather_stale_max_seconds,
            )
            run_repository_context = open_run_repository(
                backend=settings.run_store_backend.value,
                sqlite_path=settings.run_sqlite_path,
                database_url=settings.database_url,
            )
            run_repository = await run_repository_context.__aenter__()
            run_repository_entered = True
            run_coordinator = RunCoordinator(
                run_repository,
                settings.execution_budget(),
                trace_attribute_max_chars=settings.trace_attribute_max_chars,
                trace_file_dir=settings.trace_file_dir if settings.trace_file_enabled else None,
                config_values={
                    "budget": settings.execution_budget().model_dump(mode="json"),
                    "travel_provider": settings.provider.value,
                    "requirement_provider": settings.requirement_provider.value,
                    "critic_provider": settings.critic_provider.value,
                    "edit_provider": settings.edit_provider.value,
                    "weather_provider": settings.weather_provider.value,
                    "agent_mode": agent_mode.value,
                    "memory_store": settings.memory_store_backend.value,
                },
            )
            return cls(
                poi_provider=poi_provider,
                route_provider=route_provider,
                gateway=gateway,
                workflow=workflow,
                client=client,
                agent_mode=agent_mode,
                agentic_workflow=agentic_workflow,
                planner_gateway=planner_gateway,
                tool_registry=tool_registry,
                auxiliary_client=auxiliary_client,
                weather_provider=weather_provider,
                weather_gateway=weather_gateway,
                requirement_model=requirement_model,
                requirement_gateway=requirement_gateway,
                requirement_workflow=requirement_workflow,
                model_client=model_client,
                planner_model_client=planner_model_client,
                replanner_model_client=replanner_model_client,
                replanner_gateway=replanner_gateway,
                critic_model=critic_model,
                critic_gateway=critic_gateway,
                critic_model_client=critic_model_client,
                edit_model=edit_model,
                edit_gateway=edit_gateway,
                edit_model_client=edit_model_client,
                plan_repository=plan_repository,
                lifecycle_workflow=lifecycle_workflow,
                lifecycle_service=lifecycle_service,
                checkpoint_context=checkpoint_context,
                repository_context=repository_context,
                run_repository=run_repository,
                run_coordinator=run_coordinator,
                run_repository_context=run_repository_context,
                preference_repository=preference_repository,
                preference_service=preference_service,
                preference_repository_context=preference_repository_context,
                specialist_executor=specialist_executor,
            )
        except BaseException:
            if run_repository_context is not None and run_repository_entered:
                await run_repository_context.__aexit__(None, None, None)
            if repository_context is not None and repository_entered:
                await repository_context.__aexit__(None, None, None)
            if (
                preference_repository_context is not None
                and preference_repository_entered
            ):
                await preference_repository_context.__aexit__(None, None, None)
            if checkpoint_context is not None and checkpoint_entered:
                await checkpoint_context.__aexit__(None, None, None)
            if model_client is not None:
                await model_client.close()
            if planner_model_client is not None and planner_model_client is not model_client:
                await planner_model_client.close()
            if (
                replanner_model_client is not None
                and replanner_model_client is not model_client
                and replanner_model_client is not planner_model_client
            ):
                await replanner_model_client.close()
            if (
                critic_model_client is not None
                and critic_model_client is not model_client
            ):
                await critic_model_client.close()
            if (
                edit_model_client is not None
                and edit_model_client is not model_client
                and edit_model_client is not critic_model_client
            ):
                await edit_model_client.close()
            if client is not None:
                await client.aclose()
            if auxiliary_client is not None:
                await auxiliary_client.aclose()
            raise

    async def close(self) -> None:
        if self.model_client is not None:
            await self.model_client.close()
        if (
            self.planner_model_client is not None
            and self.planner_model_client is not self.model_client
        ):
            await self.planner_model_client.close()
        if (
            self.replanner_model_client is not None
            and self.replanner_model_client is not self.model_client
            and self.replanner_model_client is not self.planner_model_client
        ):
            await self.replanner_model_client.close()
        if (
            self.critic_model_client is not None
            and self.critic_model_client is not self.model_client
        ):
            await self.critic_model_client.close()
        if (
            self.edit_model_client is not None
            and self.edit_model_client is not self.model_client
            and self.edit_model_client is not self.critic_model_client
        ):
            await self.edit_model_client.close()
        if self.client is not None:
            await self.client.aclose()
        if self.auxiliary_client is not None:
            await self.auxiliary_client.aclose()
        if self.run_repository_context is not None:
            await self.run_repository_context.__aexit__(None, None, None)
        if self.repository_context is not None:
            await self.repository_context.__aexit__(None, None, None)
        if self.preference_repository_context is not None:
            await self.preference_repository_context.__aexit__(None, None, None)
        if self.checkpoint_context is not None:
            await self.checkpoint_context.__aexit__(None, None, None)

    async def plan(
        self,
        request: PlanningRequest,
        thread_id: str,
    ) -> PlanningResponse:
        return (await self.execute_plan(request, thread_id=thread_id)).payload

    async def execute_plan(
        self,
        request: PlanningRequest,
        *,
        thread_id: str,
        fault_plan: FaultPlan | None = None,
        budget: ExecutionBudget | None = None,
        run_id: str | None = None,
        principal: Principal | None = None,
        precreated: bool = False,
    ) -> ExecutionResult[PlanningResponse]:
        call = lambda: self._run_selected_planning(request, thread_id=thread_id)
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.STRUCTURED_PLAN,
            call,
            thread_id=thread_id,
            fault_plan=fault_plan,
            budget=budget,
            run_id=run_id,
            tenant_id=(principal.tenant_id if principal else "local"),
            user_id=(principal.user_id if principal else "demo"),
            precreated=precreated,
        )

    async def _run_selected_planning(
        self, request: PlanningRequest, *, thread_id: str
    ) -> PlanningResponse:
        if (
            self.agent_mode is AgentMode.DYNAMIC_PLANNER
            and self.agentic_workflow is not None
        ):
            return await run_agentic_planning(
                self.agentic_workflow, request, thread_id=thread_id
            )
        if self.agent_mode is AgentMode.SHADOW_DYNAMIC_PLANNER:
            if self.planner_gateway is None or self.tool_registry is None:
                raise RuntimeError("shadow planner is not configured")
            shadow_action = await _run_shadow_decision(
                self.planner_gateway, self.tool_registry, request, thread_id=thread_id
            )
            response = await run_planning(self.workflow, request, thread_id=thread_id)
            return response.model_copy(update={
                "agent_mode": AgentMode.SHADOW_DYNAMIC_PLANNER.value,
                "decision_count": 1,
                "action_summary": (shadow_action,),
            })
        return await run_planning(self.workflow, request, thread_id=thread_id)
    async def reserve_plan_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        principal: Principal,
        request_id: str | None = None,
    ) -> AgentRunRecord:
        if self.run_coordinator is None:
            raise RuntimeError("run coordinator is not configured")
        return await self.run_coordinator.reserve(
            RunKind.STRUCTURED_PLAN,
            run_id=run_id,
            thread_id=thread_id,
            request_id=request_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )

    async def reserve_run(
        self,
        kind: RunKind,
        *,
        run_id: str,
        thread_id: str,
        principal: Principal,
        request_id: str | None = None,
        causation_id: str | None = None,
    ) -> AgentRunRecord:
        """在流式响应返回 run_id 前创建可查询的 Run。"""
        if self.run_coordinator is None:
            raise RuntimeError("run coordinator is not configured")
        parent_run_id = (
            await self._latest_run_id(thread_id=thread_id)
            if kind is RunKind.CLARIFICATION_RESUME
            else None
        )
        return await self.run_coordinator.reserve(
            kind,
            run_id=run_id,
            thread_id=thread_id,
            request_id=request_id,
            parent_run_id=parent_run_id,
            causation_id=causation_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )

    async def plan_from_text(
        self,
        request: NaturalPlanningRequest,
        thread_id: str,
        principal: Principal | None = None,
    ) -> NaturalPlanningResponse:
        if self.requirement_workflow is None:
            raise RuntimeError("natural-language planning is not configured")
        return (
            await self.execute_plan_from_text(
                request, thread_id=thread_id, principal=principal
            )
        ).payload

    async def execute_plan_from_text(
        self,
        request: NaturalPlanningRequest,
        *,
        thread_id: str,
        principal: Principal | None = None,
        fault_plan: FaultPlan | None = None,
        budget: ExecutionBudget | None = None,
        run_id: str | None = None,
        precreated: bool = False,
    ) -> ExecutionResult[NaturalPlanningResponse]:
        if self.requirement_workflow is None:
            raise RuntimeError("natural-language planning is not configured")
        identity = principal or Principal(tenant_id="local", user_id="demo")
        call = lambda: run_natural_planning(
            self.requirement_workflow,
            request,
            thread_id=thread_id,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.NATURAL_PLAN,
            call,
            thread_id=thread_id,
            fault_plan=fault_plan,
            budget=budget,
            run_id=run_id,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
            precreated=precreated,
        )

    async def resume_from_text(
        self,
        request: ClarificationResumeRequest,
        thread_id: str,
        principal: Principal | None = None,
    ) -> NaturalPlanningResponse:
        if self.requirement_workflow is None:
            raise RuntimeError("natural-language planning is not configured")
        return (
            await self.execute_resume_from_text(
                request, thread_id=thread_id, principal=principal
            )
        ).payload

    async def execute_resume_from_text(
        self,
        request: ClarificationResumeRequest,
        *,
        thread_id: str,
        principal: Principal | None = None,
        fault_plan: FaultPlan | None = None,
        run_id: str | None = None,
        precreated: bool = False,
    ) -> ExecutionResult[NaturalPlanningResponse]:
        if self.requirement_workflow is None:
            raise RuntimeError("natural-language planning is not configured")
        identity = principal or Principal(tenant_id="local", user_id="demo")

        async def call() -> NaturalPlanningResponse:
            lock = self.resume_locks.setdefault(thread_id, asyncio.Lock())
            async with lock:
                return await resume_natural_planning(
                    self.requirement_workflow,
                    request,
                    thread_id=thread_id,
                    tenant_id=identity.tenant_id,
                    user_id=identity.user_id,
                )

        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.CLARIFICATION_RESUME,
            call,
            thread_id=thread_id,
            request_id=str(request.request_id),
            causation_id=request.interrupt_id,
            parent_run_id=(
                None
                if precreated
                else await self._latest_run_id(thread_id=thread_id)
            ),
            fault_plan=fault_plan,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
            run_id=run_id,
            precreated=precreated,
        )

    async def create_plan_session(
        self, request: PlanningRequest, *, session_id: str
    ) -> PlanSessionResponse:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return (
            await self.execute_create_plan_session(request, session_id=session_id)
        ).payload

    async def execute_create_plan_session(
        self,
        request: PlanningRequest,
        *,
        session_id: str,
        fault_plan: FaultPlan | None = None,
        principal: Principal | None = None,
    ) -> ExecutionResult[PlanSessionResponse]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        identity = principal or Principal(tenant_id="local", user_id="demo")
        call = lambda: self.lifecycle_service.create(
            request,
            session_id=session_id,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.LIFECYCLE_CREATE,
            call,
            session_id=session_id,
            thread_id=f"planning:{session_id}",
            fault_plan=fault_plan,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )

    async def create_plan_session_from_text(
        self,
        request: NaturalPlanningRequest,
        *,
        session_id: str,
        principal: Principal | None = None,
    ) -> PlanSessionResponse:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return (
            await self.execute_create_plan_session_from_text(
                request, session_id=session_id, principal=principal
            )
        ).payload

    async def execute_create_plan_session_from_text(
        self,
        request: NaturalPlanningRequest,
        *,
        session_id: str,
        principal: Principal | None = None,
        fault_plan: FaultPlan | None = None,
    ) -> ExecutionResult[PlanSessionResponse]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        identity = principal or Principal(tenant_id="local", user_id="demo")
        call = lambda: self.lifecycle_service.create_from_text(
            request,
            session_id=session_id,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.LIFECYCLE_CREATE_FROM_TEXT,
            call,
            session_id=session_id,
            thread_id=f"intake:{session_id}",
            fault_plan=fault_plan,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )

    async def execute_create_plan_session_from_checkpoint(
        self,
        *,
        thread_id: str,
        session_id: str,
        source_run_id: str,
        principal: Principal,
    ) -> ExecutionResult[PlanSessionResponse]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        call = lambda: self.lifecycle_service.create_from_checkpoint(
            thread_id,
            session_id=session_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.LIFECYCLE_CREATE,
            call,
            session_id=session_id,
            thread_id=f"lifecycle:{session_id}",
            request_id=f"selection:{source_run_id}",
            parent_run_id=source_run_id,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
        )

    async def resume_plan_session(
        self, request: LifecycleResumeRequest, *, session_id: str
    ) -> PlanSessionResponse:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return (
            await self.execute_resume_plan_session(request, session_id=session_id)
        ).payload

    async def execute_resume_plan_session(
        self,
        request: LifecycleResumeRequest,
        *,
        session_id: str,
        fault_plan: FaultPlan | None = None,
        principal: Principal | None = None,
    ) -> ExecutionResult[PlanSessionResponse]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        call = lambda: self.lifecycle_service.resume(session_id, request)
        identity = principal or Principal(tenant_id="local", user_id="demo")
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.LIFECYCLE_RESUME,
            call,
            session_id=session_id,
            thread_id=f"lifecycle:{session_id}",
            request_id=str(request.request_id),
            causation_id=request.interrupt_id,
            parent_run_id=await self._latest_run_id(session_id=session_id),
            fault_plan=fault_plan,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )

    async def get_plan_session(self, *, session_id: str) -> PlanSessionResponse:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.get(session_id)

    async def get_plan_versions(self, *, session_id: str):
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.versions(session_id)

    async def get_plan_version(self, *, session_id: str, version_id: str):
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.version(session_id, version_id)

    async def get_plan_diff(self, *, session_id: str, from_id: str, to_id: str):
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.diff(session_id, from_id, to_id)

    async def refresh_plan_weather(
        self, request: WeatherRefreshRequest, *, session_id: str
    ) -> PlanSessionResponse:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return (
            await self.execute_refresh_plan_weather(request, session_id=session_id)
        ).payload

    async def execute_refresh_plan_weather(
        self,
        request: WeatherRefreshRequest,
        *,
        session_id: str,
        fault_plan: FaultPlan | None = None,
        principal: Principal | None = None,
    ) -> ExecutionResult[PlanSessionResponse]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        call = lambda: self.lifecycle_service.refresh_weather(session_id, request)
        identity = principal or Principal(tenant_id="local", user_id="demo")
        if self.run_coordinator is None:
            return ExecutionResult(payload=await call(), run=None)
        return await self.run_coordinator.execute(
            RunKind.WEATHER_REFRESH,
            call,
            session_id=session_id,
            thread_id=f"lifecycle:{session_id}",
            request_id=str(request.request_id),
            parent_run_id=await self._latest_run_id(session_id=session_id),
            fault_plan=fault_plan,
            tenant_id=identity.tenant_id,
            user_id=identity.user_id,
        )

    async def get_plan_weather(self, *, session_id: str) -> WeatherStateView:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.weather(session_id)

    async def get_plan_weather_events(
        self, *, session_id: str
    ) -> tuple[WeatherEventView, ...]:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.weather_events(session_id)

    async def get_plan_weather_event(
        self, *, session_id: str, event_id: str
    ) -> WeatherEventView:
        if self.lifecycle_service is None:
            raise RuntimeError("plan lifecycle is not configured")
        return await self.lifecycle_service.weather_event(session_id, event_id)

    async def get_agent_run(self, run_id: str) -> AgentRunRecord:
        if self.run_repository is None:
            raise RuntimeError("run repository is not configured")
        return await self.run_repository.get(run_id)

    async def get_agent_trace(
        self, run_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> tuple[TraceEvent, ...]:
        if self.run_repository is None:
            raise RuntimeError("run repository is not configured")
        return await self.run_repository.trace(
            run_id, after_sequence=after_sequence, limit=limit
        )

    async def get_session_runs(
        self, session_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        if self.run_repository is None:
            raise RuntimeError("run repository is not configured")
        return await self.run_repository.list_for_session(session_id, limit=limit)

    async def get_thread_runs(
        self, thread_id: str, *, limit: int = 50
    ) -> tuple[AgentRunRecord, ...]:
        if self.run_repository is None:
            raise RuntimeError("run repository is not configured")
        return await self.run_repository.list_for_thread(thread_id, limit=limit)

    async def _latest_run_id(
        self, *, session_id: str | None = None, thread_id: str | None = None
    ) -> str | None:
        if self.run_repository is None:
            return None
        if session_id is not None:
            runs = await self.run_repository.list_for_session(session_id, limit=1)
        elif thread_id is not None:
            runs = await self.run_repository.list_for_thread(thread_id, limit=1)
        else:
            return None
        return runs[0].run_id if runs else None

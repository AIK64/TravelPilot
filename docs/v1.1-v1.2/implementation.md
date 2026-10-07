# v1.2 实现与使用指南

## 1. 实现结果

本次实现把 v1.1 与 v1.2 的主干能力放进同一模块化单体，但保留 `single_graph` 基线：

- `PreferenceMemoryService`：租户/用户隔离、显式偏好、Proposal 确认、冲突、TTL、撤销、删除、导出、个性化开关和敏感信息拒绝；
- Requirement Graph 的 `retrieve_relevant_preferences` Node：检索、按角色投影、预算裁剪、冲突排除，再把允许的偏好合入 `TripSpec`；
- `SpecialistExecutor`：Planner、Critic、Replanner 使用不同强类型 Context，限制字符、时间和 Handoff 次数，并记录 `agent.handoff_*` Trace；
- `TravelApplicationService`：REST、MCP 与 Worker 共用的用例入口；异步 Run 在执行前先建立持久化记录；
- `Travel MCP`：本地 stdio 和远程 Streamable HTTP，固定 MCP `2025-11-25`，提供旅行用例 Tool、受限数据 Tool 与 Run/Plan/Trace/Preference Resource；
- Provider Chain：地图 `AMap → Baidu`、天气 `AMap → QWeather`，仅对可恢复技术故障切换，带 Circuit Breaker 和 Provider Trace；
- React/Vite 前端：自然语言输入、澄清、候选、地图、Memory 和 Trace 主界面；
- PostgreSQL Plan/Run/Trace/Memory Repository、Redis Run Queue、独立 Worker 与 Docker Compose；
- 60 条跨会话 Memory Scenario 和强制消融报告脚本。

## 2. 默认本地启动

默认全部使用 Mock，不调用外部 API：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,llm-deepseek,checkpoint-sqlite]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn travel_agent.app:app --reload
```

后端文档位于 `http://127.0.0.1:8000/docs`，MCP Streamable HTTP Endpoint 是 `http://127.0.0.1:8000/mcp/`。

前端：

```powershell
Set-Location frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`。未配置浏览器地图 Key 时，规划、Memory 和 Trace 仍可使用，地图区域显示配置提示。

## 3. DeepSeek LLM

三个 LLM 职责独立配置，可以逐个启用：

```powershell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "控制台当前可用的模型 ID"

$env:REQUIREMENT_PROVIDER = "deepseek"  # 自然语言需求抽取
$env:CRITIC_PROVIDER = "deepseek"       # 软约束 Critic
$env:CRITIC_MODEL = $env:DEEPSEEK_MODEL
$env:EDIT_PROVIDER = "deepseek"         # 自然语言局部编辑
$env:EDIT_MODEL = $env:DEEPSEEK_MODEL
```

Key 只能放在环境变量或未提交的 `.env`。模型失败经过有限重试后返回结构化 Provider Failure；不会回退 Mock，也不会被伪装成 `infeasible`。

## 4. 地图与天气 API

### 4.1 AMap 主 Provider

```powershell
$env:TRAVEL_PROVIDER = "amap"
$env:WEATHER_PROVIDER = "amap"
$env:AMAP_API_KEY = "服务端 Web Service Key"
```

服务端 Key 用于 POI、路线和天气。前端 AMap JS 使用另一套绑定域名的浏览器凭证：

```powershell
$env:VITE_AMAP_JS_KEY = "浏览器 JS Key"
$env:VITE_AMAP_JS_SECURITY_CODE = "浏览器安全密钥"
```

不要把 `AMAP_API_KEY`、Baidu AK、QWeather Token 或 LLM Key 放进 `VITE_*`。

### 4.2 Baidu 地图备用

```powershell
$env:TRAVEL_PROVIDER = "amap"
$env:MAP_FALLBACK_PROVIDER = "baidu"
$env:BAIDU_MAP_AK = "百度 Web 服务 AK"
```

Baidu Adapter 使用地点检索 2.0 和轻量路线规划，并要求返回 GCJ-02，避免 Provider 原始坐标直接污染规划 State。Timeout、Connection、429、5xx 和 Schema Drift 可以切备用；认证、权限和非法参数不会切换；空列表是合法 `NO_DATA`。

### 4.3 QWeather 备用

当前 QWeather 接口使用开发者控制台提供的 API Host 与 JWT Token：

```powershell
$env:WEATHER_PROVIDER = "amap"
$env:WEATHER_FALLBACK_PROVIDER = "qweather"
$env:QWEATHER_API_HOST = "https://你的-api-host"
$env:QWEATHER_TOKEN = "你的 JWT Token"
```

实现先调用 `/geo/v2/city/lookup` 获取 Location ID 和坐标，再调用 `/weather/v1/daily/{latitude}/{longitude}`。标准化后只把 `WeatherForecast` 写回系统，不保留原始响应。

## 5. Memory API

开发环境默认主体是 `local/demo`；也可以显式传：

```text
X-Tenant-Id: tenant-a
X-User-Id: user-a
```

主要接口：

```text
GET    /api/v1/preferences
POST   /api/v1/preferences
POST   /api/v1/preferences/proposals
POST   /api/v1/preferences/proposals/{id}/confirm
POST   /api/v1/preferences/proposals/{id}/reject
PATCH  /api/v1/preferences/{id}
POST   /api/v1/preferences/{id}/revoke
DELETE /api/v1/preferences/{id}
POST   /api/v1/preferences/clear
GET    /api/v1/preferences/export
PATCH  /api/v1/profile/personalization
```

LLM 推断只能创建 Proposal，不能直接写入已确认 Memory。当前请求中的显式值始终优先；冲突类别整体排除并产生 Trace，不静默合并。

规划成功后，Requirement Graph 会继续执行 `prepare_memory_evidence → extract_preference_candidates → validate_preference_candidates → create_memory_proposals`。证据只来自用户原始需求和澄清回答，不从 Agent 自己生成的行程反推偏好；带“这次/本次”等限定的临时要求会被拒绝，其他候选最多生成 5 条待确认 Proposal。相同的已确认 Memory 或待处理 Proposal 会被去重，提取或持久化失败只把 `preference_learning_status` 标记为 `degraded`，不会把已经完成的规划改成失败。

相关 Trace 包括 `memory.extraction_started`、`memory.extraction_completed`、`memory.candidate_rejected`、`memory.candidate_deduplicated`、`memory.proposal_created` 和 `memory.extraction_degraded`。前端会在规划结果旁展示待确认提案，只有用户点击确认后才写入长期 Memory。

## 6. 异步 Run 与 SSE

```text
POST /api/v1/trips
POST /api/v1/trips/{trip_id}/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/events
POST /api/v1/runs/{run_id}/cancel
POST /api/v1/plans/from-text/stream
POST /api/v1/plans/from-text/{thread_id}/resume/stream
```

第二个请求返回 HTTP 202 和 `RunHandle`。`TraceRecorder` 产生事件后，经有序异步 Writer 逐条追加到 Run Repository；SSE 因而能在 Run 仍为 `running` 时看到 Node、Tool、Validator 和 Repair 事件，而不是等终态批量写入。终态 `finalize` 会幂等补齐完整轨迹。

`GET /runs/{run_id}/events` 使用 `Last-Event-ID` 或 `after_sequence` 续传，Run 完成后发送 `event: end`。两个自然语言流式接口在同一 SSE 中依次发送 `trace`、`result`、`end`，前端提交后即可展示实时轨迹；连接中断会取消对应的流式 Run。本地模式使用进程内任务，生产设置 `ASYNC_EXECUTION_BACKEND=redis` 后由 `travel-agent-worker` 执行结构化异步 Run。

## 7. MCP

本地 stdio：

```powershell
$env:MCP_TENANT_ID = "local"
$env:MCP_USER_ID = "demo"
$env:MCP_SCOPES = "read:data"
.\.venv\Scripts\travel-agent-mcp.exe
```

远程 MCP 与 REST 共用身份头；生产 `DEV_IDENTITY_ENABLED=false` 时必须提供可信的 `X-Tenant-Id` 和 `X-User-Id`。低层 `search_poi`、`get_route`、`get_weather` 需要 `read:data` scope。部署到自定义域名时，还必须把服务 Host 和浏览器 Origin 分别加入 `MCP_ALLOWED_HOSTS`、`MCP_ALLOWED_ORIGINS`，未命中白名单的请求会被拒绝。

MCP Tool 包含创建/恢复/取消、候选选择、变更/审批、Diff、Trace、Preference、POI、路线和天气。Resource 使用 `travel://runs/...`、`travel://plans/...` 和 `travel://users/me/preferences`，不会暴露 Chain-of-Thought、完整 Prompt 或 Provider 原始响应。

## 8. Agent 模式

```powershell
$env:AGENT_MODE = "single_graph"           # 基线
$env:AGENT_MODE = "specialist_subagents"   # 上下文隔离模式
$env:AGENT_MAX_HANDOFFS = "8"
```

Specialist 不是独立微服务。Orchestrator 把最小 Context 投影给角色，接收结构化结果，再由主 Graph 做路由、硬验证和终止。默认值应根据 `single_graph` 与 `specialist_subagents` 消融数据决定，而不是为了“多 Agent”标签机械开启。

## 9. 评测与测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\evaluate_v1_1.py
```

第二条命令读取 `evals/v1_1/memory_scenarios.jsonl` 的 60 条场景，写出 `reports/v1_1-memory-ablation.json`。真实 LLM、地图与天气 Live Smoke 默认关闭，必须显式提供凭证并接受费用。

2026-08-24 本机门禁结果：收集 565 项，563 项通过，2 项 Live Smoke 跳过；statement + branch 综合覆盖率 90.0177%。Memory 消融中，bounded Memory 的错误个性化率为 0、上下文字符数为 1026；full history 的错误个性化率为 16.67%、上下文字符数为 2056。该结果只证明固定离线数据集，不代表真实用户或真实模型质量。

## 10. Docker 生产拓扑适配

```powershell
docker compose up --build
```

Compose 包含 PostgreSQL、Redis、API/MCP、Worker 和 Web。生产必须替换数据库密码，关闭开发身份，使用 Secret Manager 注入 Key，并把 AMap JS Key 绑定到 Web 域名。PostgreSQL 首次启动执行 `infrastructure/database/001_initial.sql`；备份对象至少包括 `plan_sessions`、`agent_runs`、`trace_events`、Memory 表和 Redis AOF。

当前开发机未安装 Docker CLI，所以没有把“配置存在”冒充成“容器演练通过”。上线前仍需在目标环境执行 Compose Smoke、真实 PostgreSQL/Redis 并发、备份恢复、滚动回退和可信身份网关注入；API 不能直接信任公网客户端自行填写的 `X-Tenant-Id`/`X-User-Id`。

## 11. 与冻结设计矩阵的剩余差距

本轮已闭合 Agent 主干和可演示路径，但冻结设计中的平台强化清单大于本轮实际代码。以下内容尚不能宣称完成：PostgreSQL LangGraph Checkpointer、Alembic 升降级、OIDC/JWT Authorizer、Redis Provider Cache/限流、Outbox 天气调度、OpenTelemetry Collector、八条 Playwright 全旅程、备份恢复/多实例演练，以及 `geocode`、`reverse_geocode`、POI Detail、原生 Matrix、Weather Warning 等扩展数据工具。

这些缺口不影响当前 `Plan → Tool → Validate/Critic → Replan`、Memory、HITL、MCP 和 Failover 演示，但会影响“生产 Definition of Done”的第 7 至 10 项。后续开发必须以 [需求追踪矩阵](requirements-traceability.md) 逐项关闭，不能仅凭版本号宣称全部完成。

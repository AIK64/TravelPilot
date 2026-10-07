# v1.3 动态规划 Agent 实施说明

## 已实现

- 代码级默认、`.env.example` 与 Docker 部署示例均使用 `dynamic_planner`；`fixed_workflow` 作为显式历史基线保留，`shadow_dynamic_planner` 用于灰度对照。
- 新增强类型 `PlannerDecision` 与 Clarify、CallTool、Solve、Repair、Finish、Escalate Action。
- 新增 Action Guard，确定性校验阶段、Evidence 引用、Tool 权限和输入 Schema、候选状态、重复动作与硬约束前置条件。
- 新增 Tool Registry 与 Agent Tool Executor；模型只能选择领域工具，不能指定 Provider、URL、身份或密钥。
- 新增 Run 隔离的 Evidence Repository、Evidence Gap 派生、标准化 Observation 和上下文裁剪。
- 新增动态 LangGraph：`Observe -> Decide -> Guard -> Act -> Observe`，Tool Observation 会改变下一轮决策。
- 新增无 LangGraph 依赖的 `PlanningKernelService`。动态 Solve 直接消费 POI/Route Evidence，执行 POI 规范化、住宿锚点、CP-SAT、日程物化和 Hard Validator，不再调用固定 Planning Graph。
- `route.build_matrix` 从少量预取升级为完整规划矩阵；Solve 会核对所有必需路线键，Evidence 不完整时安全终止。
- 新增 Planner Mock/OpenAI/DeepSeek Provider 与超时、重试、预算和安全错误治理。
- 新增 Replanner Proposal、Mock/OpenAI/DeepSeek Provider 和 Repair Policy。通过 Policy 的模型 Patch 会被真实应用，再执行 Route Delta、重新物化、Hard Revalidate 和无进展检测；非法必去项删除、锁定项修改、未知引用、越界日期和重复 Proposal 会在副作用前拒绝。
- 新增无 Graph 依赖的 `QualityReviewService`，动态 Graph 显式执行 Grounded Critic、Quality Gate、Soft Repair、Route Delta、Hard Revalidate、二次评审及 baseline 恢复；软修复只有在硬约束仍合法且质量提升达到阈值时才被接受。
- `anchor.resolve`、`weather.snapshot`、`preference.retrieve` 已接入动态 `AgentToolExecutor`；`route.load_delta` 由硬修复和软修复 Kernel 在内部按变化邻接边执行，不暴露为 Planner 可任意调用的工具。
- 动态 Graph 已接入进程内 Checkpoint；Lifecycle 创建计划时通过统一 Planning Runner 读取动态 `kernel_snapshot`，不再直接调用固定工作流。
- 新增决策、Action、Observation、Evidence、Repair Proposal、Final Guard 和 No Progress Trace。
- `shadow_dynamic_planner` 只记录模型动作，不执行影子 Tool，也不修改固定工作流结果。
- 住宿仍不是必填项；未填写时复用确定性推荐住宿锚点，不让模型猜测酒店。

## 当前安全边界

- 动态路径不会直接写 Plan Repository 或 Preference Memory。
- Planner/Replanner 看不到 Provider 原始响应、Secret、用户身份字段和完整 Memory 正文。
- LLM Replanner Proposal 通过 Policy 后由 `PlanningKernelService.apply_repair()` 应用；Patch 只有在重新物化和 Hard Revalidate 后才可能进入 Finish。`REPLANNER_PROVIDER=deterministic` 仍保留确定性兼容路径，但同样经过 Proposal Policy 和复验链路。
- 规划期的自然语言缺字段继续由现有 Requirement Graph 的 LangGraph Interrupt/Resume 处理；动态内层不另建第二套澄清协议。

## 配置

```text
AGENT_MODE=fixed_workflow|dynamic_planner|shadow_dynamic_planner
PLANNER_PROVIDER=mock|openai|deepseek
PLANNER_MODEL=<explicit-model-name>
REPLANNER_PROVIDER=disabled|deterministic|mock|openai|deepseek
REPLANNER_MODEL=<explicit-model-name>
```

真实 Provider 缺少 API Key 或模型名时启动失败，不会静默回退 Mock。

## 验证

- v1.3 定向测试覆盖强类型 Action、Action Guard、Evidence、动态轨迹、实时 Trace、Repair Policy、真实 Patch 应用、Route Delta、Hard Revalidate 和 Shadow 隔离。
- 全量测试通过；保留两个既有依赖弃用/类型解析警告，不影响本次行为。

## 前端候选展示与确认

- “候选与决策”使用轻松型、均衡型、探索型卡片展示实际返回的候选数量；系统推荐方案带星标，点击卡片查看逐日行程、活动费用、交通、校验原因与方案依据。
- 未完成硬校验或校验失败的候选可查看详情，但确认按钮禁用；带警告的硬合法候选会保留提醒。后端选择节点继续拒绝非法候选。
- 页面保留自然语言规划与澄清 SSE。规划完成后，`POST /api/v1/runs/{run_id}/plan-session` 复用同一线程的 Checkpoint 创建选择 Interrupt，不再次调用模型或地图工具；先检查运行所有权，并将同一线程映射到稳定选择会话。
- 确认通过既有 `/api/v1/plan-sessions/{session_id}/resume` 提交 `select_candidate`，携带 Interrupt、Revision 和 request ID，成功后保存 V1 并显示已确认状态。浏览其他卡片不会改变确认结果。
- 偏好记忆下方的独立“原始响应 · 调试 JSON”卡片展示完整 `planning_response` 与 `selection_session`，方便检查原始规划、候选和确认结果；选择会话连接失败时可重试。
- 默认输入改成当前 Mock 解析器可识别的完整示例。以上交互仍使用当前 Provider 配置；Mock 路线在详情依据中明确标注。

## 高德接口请求节奏

- 驾车路径规划显式传入 `show_fields=cost`，确保获取解析器所需的 `cost.duration`。
- Runtime 共享的 `AMapClient` 按接口路径分别限速，同一接口两次请求的启动间隔至少为 0.4 秒；并发任务与 Gateway 重试共用此节奏，驾车与步行互不占用请求槽位。
- 限速使用单调时钟和异步锁，等待可被取消；日志 `amap.request.throttled` 记录操作名和等待秒数，不记录 Key 或请求参数。
- 限速范围是单个 Runtime 实例，适用于当前单进程后端。多个 Worker 或其他应用共用同一高德账号时，需要统一账号级限速；本地限速不协调外部流量。路线矩阵耗时会随请求数量增加，现有 Deadline 和预算继续生效。
- 定向测试覆盖驾车请求字段、并发间隔、接口独立计时和 `10021` 重试间隔，使用 HTTP MockTransport，不消耗真实高德配额。

## 景点访问去重

- 单城市行程按 POI ID 和规范化的完整名称识别重复访问目标；名称统一全半角、大小写和空白，不对相似名称做模糊合并。
- 优化器候选筛选和确定性兜底分配在排名后、截取名额前去重，保留排名最优的代表，重复记录不占用候选名额或不同日期的活动名额。
- Hard Validator 对整个候选行程检查 `duplicate_poi`，同一天或跨天的重复活动均为 Error，阻止推荐、确认或提交非法版本；住宿、休息等非活动记录不参与此检查。
- 日志 `planning.poi_duplicate_excluded` 和 `validation.duplicate_poi` 分别记录候选去重与硬校验拒绝；回归测试覆盖同 ID、同名不同 ID、缺少 ID、优化及兜底物化轨迹。

## 必去景点检索与 Planner 参数契约

- `poi.search` 接收 `city`、每次查询的 `limit` 和兼容字段 `keywords`；服务端通用查询固定使用 `keyword="景点"`、`types="110000"`，即使模型传入“自然”“人文”也不改变实际查询。兴趣保留在行程需求中参与后续匹配评分；必去景点从 `TripSpec.must_visit` 读取。
- 动态和固定工作流都先按必去名称逐个查询，再执行一次通用景点查询。例如泉州先查“大开元寺”，再查询“景点”并指定类型 `110000`。每个必去查询只采用首条结果，并检查名称是否匹配；首条不相关或为空时记录 `poi.must_visit_not_found`，后续硬校验继续检查必去约束。
- 合并时先保留必去结果，再按 POI ID 和名称匹配规则排除普通结果中的重复项。检索受查询预算、每次返回数量和候选数量限制，不表示穷尽城市全部景点；必去数量超过查询预算时明确返回错误，不静默截断。
- Tool Manifest 提供实际输入模型的完整 JSON Schema 和参数示例。Planner Prompt 明确禁止在 `poi.search.arguments` 使用 `destination`、`interests`、`must_visit`；Action Guard 把缺失字段和多余字段的错误反馈为 Observation，供下一轮模型修正。
- Agent 无进展终止时，自然语言响应允许 `failed`，保留规划的 `terminal_reason`；Run 状态和终止 Trace 同步标记失败，避免把工具或决策错误伪装成业务不可行，也避免二次响应校验异常掩盖原始原因。
- 定向测试覆盖分阶段查询、首条结果筛选、去重、预算、Schema 一致性、模型参数纠正与持续错误后的失败轨迹。测试使用 Mock Provider，不代表真实高德排序或真实模型效果。

## 后续工作

动态生产路径已经解除对固定 Planning Graph 的运行时依赖。固定路径只保留为 Baseline、Shadow 主结果和历史消融评测对象；旧 `specialist_subagents` / `shadow_subagents` 配置会分别映射到 `dynamic_planner` / `shadow_dynamic_planner`，Runtime 不再装配旧 `SpecialistExecutor`。

后续工作集中在可靠性和评测，而不是功能迁移：为动态 Evidence/Checkpoint 增加跨进程持久化，建立 30～50 条动态轨迹数据集，并在获得真实 Provider 配置后运行真实模型 Profile。未完成这些 Gate 前，不把固定 Mock Benchmark 指标表述为新动态链路的线上效果。

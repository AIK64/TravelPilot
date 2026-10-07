import type { PlanChangeAction, PlanSession, PlanningResult } from "./plan";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export type AgentResponse = {
  status: string;
  thread_id?: string;
  session_id?: string;
  interrupt?: { id: string; payload: { questions?: string[] } };
  selected_plan?: unknown;
  candidates?: unknown[];
  preference_learning_status?: string;
  memory_proposals?: MemoryProposal[];
  planning?: PlanningResult | null;
  [key: string]: unknown;
};

export type TraceEvent = {
  event_id: string;
  sequence: number;
  event_type: string;
  status: string;
  node?: string;
  operation?: string;
  duration_ms?: number;
  attributes: Record<string, string | number | boolean | null>;
};

const identityHeaders = {
  "X-Tenant-Id": import.meta.env.VITE_DEV_TENANT_ID ?? "local",
  "X-User-Id": import.meta.env.VITE_DEV_USER_ID ?? "demo",
};

async function request<T>(path: string, init?: RequestInit): Promise<{ data: T; response: Response }> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...identityHeaders, ...init?.headers },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data?.detail?.message ?? (typeof data?.detail === "string" ? data.detail : null) ?? data?.message ?? `HTTP ${response.status}`);
  return { data, response };
}

export async function planFromText(query: string) {
  return request<AgentResponse>("/api/v1/plans/from-text", {
    method: "POST",
    body: JSON.stringify({ text: query }),
  });
}

export async function resumePlan(threadId: string, interruptId: string, answer: string) {
  return request<AgentResponse>(`/api/v1/plans/from-text/${threadId}/resume`, {
    method: "POST",
    body: JSON.stringify({
      interrupt_id: interruptId,
      request_id: crypto.randomUUID(),
      answer,
    }),
  });
}

type PlanStreamOptions = {
  signal?: AbortSignal;
  onStarted?: (runId: string, threadId: string) => void;
  onEvent: (event: TraceEvent) => void;
};

export function streamPlanFromText(query: string, options: PlanStreamOptions) {
  return consumePlanStream(
    "/api/v1/plans/from-text/stream",
    { text: query },
    options,
  );
}

export function streamResumePlan(
  threadId: string,
  interruptId: string,
  answer: string,
  options: PlanStreamOptions,
) {
  return consumePlanStream(
    `/api/v1/plans/from-text/${threadId}/resume/stream`,
    { interrupt_id: interruptId, request_id: crypto.randomUUID(), answer },
    options,
  );
}

async function consumePlanStream(
  path: string,
  body: object,
  { signal, onStarted, onEvent }: PlanStreamOptions,
): Promise<{ data: AgentResponse; response: Response }> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...identityHeaders,
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail?.message ?? `Plan stream HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("当前浏览器不支持流式响应");

  const runId = response.headers.get("X-Agent-Run-Id");
  const threadId = response.headers.get("X-Agent-Thread-Id");
  if (!runId || !threadId) throw new Error("流式响应缺少 Run 标识");
  onStarted?.(runId, threadId);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: AgentResponse | null = null;

  const consumeBlock = (block: string) => {
    const parsed = parseSseBlock(block);
    if (parsed.event === "trace" && parsed.data) {
      onEvent(JSON.parse(parsed.data) as TraceEvent);
    } else if (parsed.event === "result" && parsed.data) {
      result = JSON.parse(parsed.data) as AgentResponse;
    } else if (parsed.event === "error") {
      const error = parsed.data ? JSON.parse(parsed.data) : null;
      throw new Error(error?.message ?? "Agent 运行失败");
    }
    return parsed.event === "end";
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (consumeBlock(block)) {
        if (!result) throw new Error("Trace 流结束前未返回规划结果");
        return { data: result, response };
      }
      boundary = buffer.indexOf("\n\n");
    }
    if (done) {
      if (buffer.trim()) consumeBlock(buffer);
      if (!result) throw new Error("Trace 流意外中断");
      return { data: result, response };
    }
  }
}

function parseSseBlock(block: string) {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return { event, data: data.join("\n") };
}

export async function getTrace(runId: string, after = 0) {
  return request<{ events: TraceEvent[] }>(`/api/v1/runs/${runId}/trace?after_sequence=${after}&limit=500`);
}

type TraceStreamOptions = {
  after?: number;
  signal?: AbortSignal;
  onEvent: (event: TraceEvent) => void;
};

/**
 * 使用 fetch 而不是原生 EventSource：当前 API 通过自定义 Header 传递租户与用户身份，
 * EventSource 无法附带这些 Header。
 */
export async function streamTrace(
  runId: string,
  { after = 0, signal, onEvent }: TraceStreamOptions,
) {
  const response = await fetch(
    `${API_BASE}/api/v1/runs/${runId}/events?after_sequence=${after}`,
    { headers: { ...identityHeaders, Accept: "text/event-stream" }, signal },
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail?.message ?? `Trace stream HTTP ${response.status}`);
  }
  if (!response.body) throw new Error("当前浏览器不支持流式响应");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const consumeBlock = (block: string) => {
    const parsed = parseSseBlock(block);
    if (parsed.event === "trace" && parsed.data) {
      onEvent(JSON.parse(parsed.data) as TraceEvent);
    }
    return parsed.event === "end";
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (consumeBlock(block)) return;
      boundary = buffer.indexOf("\n\n");
    }
    if (done) {
      if (buffer.trim()) consumeBlock(buffer);
      return;
    }
  }
}

/** 复用已完成 Run 的候选和 Checkpoint，进入用户选择阶段，不重新规划。 */
export function createSelectionSession(runId: string) {
  return request<PlanSession>(`/api/v1/runs/${runId}/plan-session`, { method: "POST" });
}

export function confirmCandidate(session: PlanSession, candidateId: string) {
  return resumePlanSession(session, { kind: "select_candidate", candidate_id: candidateId });
}

export function resumePlanSession(
  session: PlanSession,
  action: PlanChangeAction | { kind: "select_candidate"; candidate_id: string },
  signal?: AbortSignal,
) {
  if (!session.interrupt) throw new Error("计划会话已过期，请重新生成计划");
  if (!session.allowed_actions.includes(action.kind)) throw new Error("当前计划状态不允许此操作");
  return request<PlanSession>(`/api/v1/plan-sessions/${session.session_id}/resume`, {
    method: "POST",
    signal,
    body: JSON.stringify({
      interrupt_id: session.interrupt.id,
      request_id: crypto.randomUUID(),
      expected_session_revision: session.session_revision,
      expected_active_version_id: session.active_version?.version_id ?? null,
      action,
    }),
  });
}

export async function getPreferences() {
  return request<{ items: Preference[]; personalization: { enabled: boolean; revision: number } }>("/api/v1/preferences");
}

export type Preference = {
  memory_id: string;
  category: string;
  value: unknown;
  confidence: number;
  confirmation_status: string;
  revision: number;
  revoked_at?: string;
};

export type MemoryProposal = {
  proposal_id: string;
  category: string;
  value: unknown;
  confidence: number;
  reason: string;
  status: string;
};

export async function confirmMemoryProposal(proposalId: string) {
  return request<Preference>(`/api/v1/preferences/proposals/${proposalId}/confirm`, {
    method: "POST",
    body: JSON.stringify({ request_id: crypto.randomUUID() }),
  });
}

export async function rejectMemoryProposal(proposalId: string) {
  return request<MemoryProposal>(`/api/v1/preferences/proposals/${proposalId}/reject`, {
    method: "POST",
    body: JSON.stringify({ request_id: crypto.randomUUID() }),
  });
}

export async function createPreference(category: string, value: unknown) {
  return request<Preference>("/api/v1/preferences", {
    method: "POST",
    body: JSON.stringify({ category, value, scope: "global" }),
  });
}

export async function deletePreference(memoryId: string) {
  return request<void>(`/api/v1/preferences/${memoryId}`, { method: "DELETE" });
}

export async function setPersonalization(enabled: boolean, revision: number) {
  return request(`/api/v1/profile/personalization`, {
    method: "PATCH",
    body: JSON.stringify({ enabled, expected_revision: revision }),
  });
}

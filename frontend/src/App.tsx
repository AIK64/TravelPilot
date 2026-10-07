import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { AgentResponse, confirmCandidate, createSelectionSession, getTrace, resumePlanSession, streamPlanFromText, streamResumePlan, TraceEvent } from "./api";
import { MapPanel } from "./components/MapPanel";
import { PreferencePanel } from "./components/PreferencePanel";
import { TracePanel } from "./components/TracePanel";
import { PlanPanel } from "./components/PlanPanel";
import { PlanAdjustPanel } from "./components/PlanAdjustPanel";
import { collectPlanPoints, passesHardValidation, PlanChangeAction, PlanSession } from "./plan";

const example = "2026年10月2日到10月4日去杭州，3个人，2日10:30到达杭州东站，4日19:00从杭州东站离开，住西湖东侧，预算1500元，喜欢自然和人文，灵隐寺必须去，不想太累。";

export default function App() {
  const [query, setQuery] = useState(example);
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<AgentResponse | null>(null);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [traceStreaming, setTraceStreaming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [session, setSession] = useState<PlanSession | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [sourceRunId, setSourceRunId] = useState<string | null>(null);
  const [focusedId, setFocusedId] = useState<string>();
  const [editText, setEditText] = useState("");
  const [editError, setEditError] = useState<string | null>(null);
  const traceController = useRef<AbortController | null>(null);
  const candidates = useMemo(() => session?.active_version
    ? [session.active_version.candidate]
    : session?.candidates ?? result?.planning?.candidates ?? [], [session, result]);
  const showingPreview = !!session?.pending_preview && (session.status === "awaiting_change_approval"
    || (session.status === "change_rejected" && session.pending_preview.status === "invalid"));
  const focused = showingPreview ? session?.pending_preview?.candidate
    : session?.active_version?.candidate ?? candidates.find(candidate => candidate.id === focusedId);
  const points = useMemo(() => collectPlanPoints(focused), [focused]);

  useEffect(() => () => traceController.current?.abort(), []);

  function startTrace() {
    traceController.current?.abort();
    const controller = new AbortController();
    traceController.current = controller;
    setEvents([]);
    setTraceStreaming(true);
    return controller;
  }

  function appendTrace(incoming: TraceEvent) {
    setEvents(current => {
      if (current.some(event => event.sequence === incoming.sequence)) return current;
      return [...current, incoming].sort((left, right) => left.sequence - right.sequence);
    });
  }

  async function finishTrace(controller: AbortController, fallbackRunId: string | null) {
    if (fallbackRunId && !controller.signal.aborted) {
      const snapshot = await getTrace(fallbackRunId).catch(() => null);
      if (snapshot) setEvents(snapshot.data.events);
    }
    if (traceController.current === controller) {
      traceController.current = null;
      setTraceStreaming(false);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError(null);
    setResult(null); setSession(null); setSelectionError(null); setFocusedId(undefined);
    setSourceRunId(null); setThreadId(null); setRunId(null); setAnswer("");
    setEditText(""); setEditError(null);
    const controller = startTrace();
    let activeRunId: string | null = null;
    try {
      const { data } = await streamPlanFromText(query, {
        signal: controller.signal,
        onStarted: (id, currentThreadId) => {
          activeRunId = id; setRunId(id); setThreadId(currentThreadId);
        },
        onEvent: appendTrace,
      });
      setResult(data); setThreadId(data.thread_id ?? null);
      await prepareSelection(data, activeRunId);
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      await finishTrace(controller, activeRunId); setBusy(false);
    }
  }

  async function clarify() {
    if (!threadId) return; setBusy(true); setError(null);
    const controller = startTrace();
    let activeRunId: string | null = null;
    try {
      if (!result?.interrupt) return;
      const { data } = await streamResumePlan(threadId, result.interrupt.id, answer, {
        signal: controller.signal,
        onStarted: (id) => { activeRunId = id; setRunId(id); },
        onEvent: appendTrace,
      });
      setResult(data);
      setAnswer("");
      await prepareSelection(data, activeRunId);
    } catch (reason) {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      await finishTrace(controller, activeRunId); setBusy(false);
    }
  }

  async function prepareSelection(data: AgentResponse, planningRunId: string | null) {
    if (data.status !== "completed" || !data.planning?.candidates.length) return;
    setSourceRunId(planningRunId);
    setFocusedId(data.planning.selected_plan?.id ?? data.planning.candidates[0]?.id);
    try {
      if (!planningRunId) throw new Error("缺少规划运行标识，无法创建选择会话");
      const { data: selection } = await createSelectionSession(planningRunId);
      setSession(selection); setSelectionError(null);
    } catch (reason) {
      setSelectionError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function retrySelection() {
    if (!result || busy) return;
    setBusy(true);
    try { await prepareSelection(result, sourceRunId); }
    finally { setBusy(false); }
  }

  async function chooseCandidate(candidateId: string) {
    const candidate = candidates.find(item => item.id === candidateId);
    if (busy || !session || !session.allowed_actions.includes("select_candidate") || !candidate || !passesHardValidation(candidate)) return;
    setBusy(true); setError(null);
    const controller = startTrace();
    let selectionRunId: string | null = null;
    try {
      const { data, response } = await confirmCandidate(session, candidateId);
      setSession(data); setFocusedId(data.active_version?.candidate.id ?? candidateId);
      setEditText(""); setEditError(null);
      selectionRunId = response.headers.get("X-Agent-Run-Id");
      setRunId(selectionRunId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      await finishTrace(controller, selectionRunId); setBusy(false);
    }
  }

  async function changePlan(action: PlanChangeAction) {
    if (busy || !session || !session.allowed_actions.includes(action.kind)) return;
    setBusy(true); setEditError(null);
    const controller = startTrace();
    let changeRunId: string | null = null;
    try {
      const { data, response } = await resumePlanSession(session, action, controller.signal);
      setSession(data); setFocusedId(data.active_version?.candidate.id);
      if (data.status === "awaiting_change_approval" || action.kind === "approve_preview") setEditText("");
      changeRunId = response.headers.get("X-Agent-Run-Id");
      setRunId(changeRunId);
    } catch (reason) {
      if (!controller.signal.aborted) setEditError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      await finishTrace(controller, changeRunId); setBusy(false);
    }
  }

  return <main>
    <header><div className="brand-mark">TA</div><div><p>AGENT ENGINEERING STUDIO</p><h1>Travel Agent</h1></div><span className="status"><i /> graph online</span></header>
    <div className="hero"><div><span className="eyebrow">Plan → Tool Use → Validate → Replan</span><h2>把旅行约束，变成可验证的计划。</h2><p>偏好记忆、工具事实、Critic 与局部重规划都留在可观察轨迹里。</p></div><div className="hero-stat"><strong>{events.length}</strong><span>TRACE EVENTS</span></div></div>
    <div className="workspace">
      <div className="left-column">
        <section className="panel composer"><div className="panel-title"><span>旅行需求</span><small>自然语言</small></div>
          <form onSubmit={submit}><textarea aria-label="旅行需求" disabled={busy} value={query} onChange={e => setQuery(e.target.value)} /><div className="composer-actions"><span>Memory 会在 Graph 内按相关性裁剪</span><button disabled={busy || !query.trim()}>{busy ? "Agent 运行中…" : "生成旅行计划 →"}</button></div></form>
          {error && <p className="error">{error}</p>}
          {result?.status === "needs_clarification" && result.interrupt && <div className="clarify"><strong>需要你补充</strong><p>{(result.interrupt.payload.questions ?? []).join("；")}</p><div><input aria-label="补充旅行信息" disabled={busy} value={answer} onChange={e => setAnswer(e.target.value)} /><button disabled={busy || !answer.trim()} onClick={() => void clarify()}>继续运行</button></div></div>}
        </section>
        <PlanPanel
          candidates={candidates}
          recommendedId={result?.planning?.selected_plan?.id}
          confirmedId={session?.active_version?.candidate.id}
          focusedId={focusedId}
          onFocus={setFocusedId}
          onConfirm={id => void chooseCandidate(id)}
          canConfirm={!!session?.allowed_actions.includes("select_candidate")}
          busy={busy}
          status={result?.status === "needs_clarification" ? "等待补充" : "等待输入"}
          message={result?.message as string | undefined}
          selectionError={selectionError}
          onRetry={() => void retrySelection()}
          detailCandidate={focused}
          versionNumber={session?.active_version?.number}
          showingPreview={showingPreview}
          adjustment={session?.active_version && <PlanAdjustPanel session={session} text={editText} onTextChange={setEditText} onAction={action => void changePlan(action)} busy={busy} error={editError} />}
        />
        <PreferencePanel
          proposals={result?.memory_proposals}
          onProposalResolved={proposalId => setResult(current => current ? {
            ...current,
            memory_proposals: (current.memory_proposals ?? []).filter(item => item.proposal_id !== proposalId),
          } : current)}
        />
        <section className="panel debug-panel"><div className="panel-title"><span>原始响应 · 调试 JSON</span><small>规划与选择会话</small></div>
          {result ? <pre tabIndex={0} aria-label="原始响应 JSON">{JSON.stringify({ planning_response: result, selection_session: session }, null, 2)}</pre> : <p className="empty">运行后展示完整规划响应及候选确认结果，方便调试观察。</p>}
        </section>
      </div>
      <div className="right-column"><MapPanel points={points} /><TracePanel events={events} streaming={traceStreaming} />{runId && <div className="run-chip">run_id <code>{runId}</code></div>}</div>
    </div>
  </main>;
}

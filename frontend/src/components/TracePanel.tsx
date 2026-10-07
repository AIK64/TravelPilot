import type { TraceEvent } from "../api";

function stageOf(event: TraceEvent) {
  const type = event.event_type;
  const node = event.node ?? "";
  if (type.startsWith("tool.") || type.startsWith("provider.")) return "Tool Use";
  if (type.startsWith("validation.") || node.includes("validate") || node.includes("critic")) return "Validate";
  if (type.startsWith("repair.") || node.includes("repair")) return "Replan";
  return "Plan";
}

export function TracePanel({ events, streaming = false }: { events: TraceEvent[]; streaming?: boolean }) {
  return <section className="panel trace-panel">
    <div className="panel-title"><span>执行轨迹</span><small>{streaming ? "LIVE · " : ""}{events.length} events</small></div>
    {events.length === 0 ? <p className="empty">{streaming ? "已连接 Trace 流，等待 Agent 事件…" : "提交规划后，这里会展示 Node、Tool、Handoff 与降级事件。"}</p> :
      <ol className="timeline">{events.map(event => <li key={event.event_id}>
        <span className={`event-dot ${event.status}`} />
        <div><strong><em>{stageOf(event)}</em>{event.event_type}</strong><small>#{event.sequence} · {event.node ?? event.operation ?? "run"}{event.duration_ms != null ? ` · ${event.duration_ms}ms` : ""}</small></div>
      </li>)}</ol>}
  </section>;
}

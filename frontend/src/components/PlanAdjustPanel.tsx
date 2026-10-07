import { FormEvent } from "react";
import type { PlanChangeAction, PlanSession } from "../plan";

type Props = {
  session: PlanSession;
  text: string;
  onTextChange: (text: string) => void;
  onAction: (action: PlanChangeAction) => void;
  busy: boolean;
  error: string | null;
};

export function PlanAdjustPanel({ session, text, onTextChange, onAction, busy, error }: Props) {
  const preview = session.pending_preview;
  const approval = session.allowed_actions.includes("approve_preview");
  const canEdit = !!session.interrupt && session.allowed_actions.includes("edit_text");
  const token = session.interrupt?.payload.approval_token;
  const clarification = session.allowed_actions.includes("clarify_edit");
  const validPreview = !!preview && preview.hard_validation.valid
    && !preview.hard_validation.violations.some(issue => issue.severity === "error");

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!busy && canEdit && text.trim()) onAction({ kind: "edit_text", text: text.trim() });
  }

  return <section className="panel plan-adjust" aria-label="计划微调">
    <div className="panel-title"><span>计划微调</span><small>{busy ? "处理中…" : approval ? "等待确认修改" : clarification ? "需要补充" : `当前版本 V${session.active_version?.number ?? 1}`}</small></div>
    <div className="adjust-body">
      <p className="adjust-hint">用自然语言告诉我你想怎么调整，例如“把开元寺移到第二天”或“删除清源山”。</p>
      <form className="adjust-form" onSubmit={submit}>
        <input aria-label="计划微调需求" maxLength={2000} value={text} onChange={event => onTextChange(event.target.value)} disabled={busy || !canEdit} placeholder="输入希望修改的景点、日期或游览顺序…" />
        <button type="submit" disabled={busy || !canEdit || !text.trim()}>{busy ? "处理中…" : "提交"}</button>
      </form>
      {error && <p className="adjust-error" role="alert">{error}</p>}
      {session.message && <p className="adjust-message" role="status">{session.message}</p>}
      {session.status === "change_rejected" && preview?.status === "invalid" && <div className="adjust-error" role="alert">
        <strong>微调未通过校验，原计划仍保留</strong>
        <ul>{preview.hard_validation.violations.filter(issue => issue.severity === "error").map((issue, index) => <li key={index}>{issue.message}{issue.repair_hint && `；${issue.repair_hint}`}</li>)}</ul>
        <p>下方为未保存的修改预览，请调整输入后重新提交。</p>
      </div>}
      {clarification && <div className="adjust-clarification">
        <strong>请确认要修改哪一个活动</strong><p>点击对应活动后继续微调。</p>
        <div className="adjust-actions">{(session.interrupt?.payload.items ?? []).map(item => <button type="button" className="secondary" key={item.item_id} disabled={busy} onClick={() => onAction({ kind: "clarify_edit", item_id: item.item_id })}>{item.date ?? item.day} · {item.name}</button>)}</div>
      </div>}
      {approval && preview && <div className="adjust-preview">
        <strong>修改预览</strong><p>下方详细行程正在展示修改后的安排，确认后保存为新版本。</p>
        <ul>
          {preview.diff.added_items.map((item, index) => <li key={`add-${index}`}>新增：{item.name}{item.to_date && `（${item.to_date}）`}</li>)}
          {preview.diff.removed_items.map((item, index) => <li key={`remove-${index}`}>删除：{item.name}</li>)}
          {preview.diff.moved_items.map((item, index) => <li key={`move-${index}`}>移动：{item.name} · {item.from_date} → {item.to_date}</li>)}
          {preview.diff.reordered_items.map((item, index) => <li key={`order-${index}`}>调整顺序：{item.name}</li>)}
          {preview.diff.time_changes.length > 0 && <li>重新计算了 {preview.diff.time_changes.length} 项活动时间。</li>}
        </ul>
        {!validPreview && <p className="adjust-error">修改尚未通过硬校验，请查看下方原因。</p>}
        <div className="adjust-actions">
          <button type="button" disabled={busy || !validPreview || !token} onClick={() => token && onAction({ kind: "approve_preview", preview_id: preview.preview_id, approval_token: token })}>确认修改</button>
          <button type="button" className="secondary" disabled={busy || !session.allowed_actions.includes("reject_preview")} onClick={() => onAction({ kind: "reject_preview", preview_id: preview.preview_id })}>保留原计划</button>
        </div>
      </div>}
    </div>
  </section>;
}

import { FormEvent, useEffect, useState } from "react";
import { confirmMemoryProposal, createPreference, deletePreference, getPreferences, MemoryProposal, Preference, rejectMemoryProposal, setPersonalization } from "../api";

const categoryLabels: Record<string, string> = {
  pace: "行程节奏", preferred_categories: "偏好类别", avoided_categories: "避开类别",
  walking_tolerance: "步行容忍度", food_preferences: "饮食偏好", preferred_transport: "交通偏好",
  schedule_preferences: "时间偏好", accessibility_needs: "无障碍需求", budget_style: "预算偏好",
};
const paceLabels: Record<string, string> = { relaxed: "轻松", balanced: "均衡", exploration: "探索" };

function displayValue(value: unknown, category: string): string {
  if (Array.isArray(value)) return value.map(item => displayValue(item, category)).join("、");
  if (value !== null && typeof value === "object") {
    return Object.entries(value).map(([key, item]) => `${categoryLabels[key] ?? key}：${displayValue(item, key)}`).join("；");
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  const text = String(value ?? "未填写");
  return category === "pace" ? paceLabels[text] ?? text : text;
}

export function PreferencePanel({ proposals = [], onProposalResolved }: { proposals?: MemoryProposal[]; onProposalResolved?: (proposalId: string) => void }) {
  const [items, setItems] = useState<Preference[]>([]);
  const [enabled, setEnabled] = useState(true);
  const [revision, setRevision] = useState(0);
  const [category, setCategory] = useState("pace");
  const [value, setValue] = useState("relaxed");
  const [pendingProposal, setPendingProposal] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const { data } = await getPreferences();
    setItems(data.items); setEnabled(data.personalization.enabled); setRevision(data.personalization.revision);
  }
  useEffect(() => { void refresh(); }, []);
  async function submit(event: FormEvent) {
    event.preventDefault(); await createPreference(category, value); await refresh();
  }
  async function toggle() {
    const { data } = await setPersonalization(!enabled, revision) as { data: { enabled: boolean; revision: number } };
    setEnabled(data.enabled); setRevision(data.revision);
  }
  async function decide(proposalId: string, accept: boolean) {
    setPendingProposal(proposalId); setError(null);
    try {
      if (accept) await confirmMemoryProposal(proposalId);
      else await rejectMemoryProposal(proposalId);
      onProposalResolved?.(proposalId);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "偏好操作失败，请重试");
    } finally {
      setPendingProposal(null);
    }
  }
  return <section className="panel memory-panel">
    <div className="panel-title"><span>偏好记忆</span><button className={`switch ${enabled ? "on" : ""}`} onClick={() => void toggle()}>{enabled ? "个性化开启" : "个性化关闭"}</button></div>
    <form className="memory-form" onSubmit={submit}>
      <select value={category} onChange={e => setCategory(e.target.value)}>
        <option value="pace">行程节奏</option><option value="preferred_categories">偏好类别</option><option value="avoided_categories">避开类别</option><option value="walking_tolerance">步行容忍度</option><option value="food_preferences">饮食偏好</option>
      </select>
      {category === "pace" ? <select value={value} onChange={e => setValue(e.target.value)} aria-label="偏好值">
        {!paceLabels[value] && <option value={value}>{value || "请选择节奏"}</option>}
        {Object.entries(paceLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select> : <input value={value} onChange={e => setValue(e.target.value)} aria-label="偏好值" />}
      <button type="submit">保存显式偏好</button>
    </form>
    {proposals.length > 0 && <div className="memory-proposals">
      <div className="memory-section-heading"><strong>是否记住这些旅行偏好？</strong><span>确认后用于后续行程推荐</span></div>
      <div className="memory-proposal-grid">{proposals.map(proposal => <article className="memory-proposal-card" key={proposal.proposal_id}>
        <div className="memory-proposal-heading"><strong>{categoryLabels[proposal.category] ?? proposal.category}</strong><span className="memory-confidence">置信度 {Math.round(proposal.confidence * 100)}%</span></div>
        <p className="memory-proposal-value">{displayValue(proposal.value, proposal.category)}</p>
        <p className="memory-proposal-reason">{proposal.reason}</p>
        <div className="memory-proposal-actions"><button className="memory-confirm" disabled={pendingProposal !== null} onClick={() => void decide(proposal.proposal_id, true)}>{pendingProposal === proposal.proposal_id ? "处理中…" : "确认记住"}</button><button className="memory-ignore" disabled={pendingProposal !== null} onClick={() => void decide(proposal.proposal_id, false)}>忽略</button></div>
      </article>)}</div>
    </div>}
    {error && <p className="error" role="alert">{error}</p>}
    <div className="memory-list">{items.map(item => <article key={item.memory_id}>
      <div><strong>{categoryLabels[item.category] ?? item.category}</strong><span>{displayValue(item.value, item.category)}</span></div>
      <small>{({ confirmed: "已确认", pending: "待确认", rejected: "已忽略" } as Record<string, string>)[item.confirmation_status] ?? item.confirmation_status} · 置信度 {Math.round(item.confidence * 100)}%</small>
      <button onClick={() => void deletePreference(item.memory_id).then(refresh)}>删除</button>
    </article>)}</div>
  </section>;
}

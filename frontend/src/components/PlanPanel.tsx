import type { PlanCandidate, PlanViolation } from "../plan";
import type { ReactNode } from "react";
import { passesHardValidation } from "../plan";

const styles: Record<string, { title: string; subtitle: string; icon: string }> = {
  relaxed: { title: "轻松型", subtitle: "留出休息时间，从容游览", icon: "01" },
  balanced: { title: "均衡型", subtitle: "兼顾体验丰富度与行程节奏", icon: "02" },
  exploration: { title: "探索型", subtitle: "体验更多地点与不同主题", icon: "03" },
};

export function styleName(style: string) { return styles[style]?.title ?? style; }
const money = (value: string | number | null | undefined) => value == null ? "待确认" : `¥${Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 2 })}`;
const distance = (meters: number) => `${(meters / 1000).toFixed(1)} km`;
const time = (value: string) => value.slice(11, 16);

function validationLabel(candidate: PlanCandidate) {
  if (!candidate.validation) return "尚未完成硬校验";
  if (!passesHardValidation(candidate)) return "硬校验未通过";
  return candidate.validation.status === "valid_with_warnings" ? "硬校验通过 · 有提醒" : "硬校验通过";
}

function Issues({ issues }: { issues: PlanViolation[] }) {
  return <ul className="plan-issues">{issues.map((issue, index) => <li key={`${issue.type}-${index}`} className={issue.severity === "error" ? "blocking" : "warning"}>
    <span>{issue.severity === "error" ? "未通过" : "提醒"}</span>
    <div>{issue.day && <small>{issue.day} · </small>}{issue.message}{issue.repair_hint && <small className="repair-hint">{issue.repair_hint}</small>}</div>
  </li>)}</ul>;
}

type Props = {
  candidates: PlanCandidate[];
  recommendedId?: string;
  confirmedId?: string;
  focusedId?: string;
  onFocus: (id: string) => void;
  onConfirm: (id: string) => void;
  canConfirm: boolean;
  busy: boolean;
  status: string;
  message?: string | null;
  selectionError?: string | null;
  onRetry: () => void;
  adjustment?: ReactNode;
  detailCandidate?: PlanCandidate;
  versionNumber?: number;
  showingPreview?: boolean;
};

export function PlanPanel({ candidates, recommendedId, confirmedId, focusedId, onFocus, onConfirm, canConfirm, busy, status, message, selectionError, onRetry, adjustment, detailCandidate, versionNumber, showingPreview }: Props) {
  const focused = detailCandidate ?? candidates.find(candidate => candidate.id === focusedId);
  const hasValidCandidate = candidates.some(passesHardValidation);
  return <><section className="panel plan-panel">
    <div className="panel-title"><span>候选与决策</span><small>{busy ? "处理中…" : confirmedId ? "已确认计划" : candidates.length ? (hasValidCandidate ? "等待选择" : "无可确认方案") : status}</small></div>
    {candidates.length === 0
      ? <p className="empty">{message || (busy ? "正在生成候选方案，执行过程可在右侧轨迹中查看。" : "生成计划后，在这里比较候选方案并确认选择。")}</p>
      : <>
        <div className="plan-intro"><p>{confirmedId ? "当前已确认计划" : `共 ${candidates.length} 个候选方案`}</p><span>{confirmedId ? "可在下方输入微调需求，调整景点、日期或游览顺序。" : hasValidCandidate ? "点击卡片查看行程，确认后保存为正式计划。" : "当前候选均未通过硬校验，可点击查看原因，请调整需求后重新生成。"}</span></div>
        {confirmedId && <p className="selection-notice" role="status">✓ 已确认{styleName(candidates.find(item => item.id === confirmedId)?.style ?? "")} · 当前版本 V{versionNumber ?? 1}</p>}
        {selectionError && <div className="selection-error" role="alert"><span>{selectionError}</span><button type="button" disabled={busy} onClick={onRetry}>重试连接选择会话</button></div>}
        <div className={`candidate-grid ${confirmedId ? "single-candidate" : ""}`}>{candidates.map(candidate => {
          const preset = styles[candidate.style];
          const valid = passesHardValidation(candidate);
          const confirmed = candidate.id === confirmedId;
          const isFocused = candidate.id === focusedId;
          return <article key={candidate.id} className={`candidate-card ${isFocused ? "is-focused" : ""} ${confirmed ? "is-confirmed" : ""} ${!valid ? "is-invalid" : ""}`}>
            <button type="button" className="candidate-open" onClick={() => onFocus(candidate.id)} aria-expanded={isFocused} aria-controls="plan-detail" aria-label={`查看${styleName(candidate.style)}计划详情`}>
              <div className="candidate-heading"><span className="candidate-number">{preset?.icon ?? "·"}</span>{candidate.id === recommendedId && <span className="recommended-badge"><span aria-hidden="true">★</span> 系统推荐</span>}</div>
              <h3>{styleName(candidate.style)}</h3><p className="candidate-subtitle">{preset?.subtitle ?? "查看每日安排与约束校验"}</p>
              <div className="candidate-stats"><div><small>已知活动费用</small><strong>{money(candidate.metrics.known_estimated_cost)}</strong></div><div><small>交通时间</small><strong>{candidate.metrics.total_travel_minutes}<em> 分钟</em></strong></div></div>
              <p className="candidate-places">{candidate.days.flatMap(day => day.items.map(item => item.name)).join(" · ") || "暂无活动安排"}</p>
              <span className={`validation-badge ${valid ? "passed" : "failed"}`}>{valid ? "✓" : "!"} {validationLabel(candidate)}</span>
              {!valid && <ul className="candidate-failures">{(candidate.validation?.violations.filter(issue => issue.severity === "error") ?? []).map((issue, index) => <li key={index}>{issue.message}</li>)}{!candidate.validation && <li>缺少校验结果，暂不能确认。</li>}</ul>}
              <span className="view-detail">{isFocused ? "正在查看详情 ↓" : "查看详细行程 →"}</span>
            </button>
            <div className="candidate-footer"><button type="button" className="confirm-plan" disabled={busy || !valid || !canConfirm || !!confirmedId} onClick={() => onConfirm(candidate.id)}>
              {confirmed ? "✓ 已确认选择" : !valid ? "暂不可确认" : confirmedId ? "已确认其他方案" : busy ? "处理中…" : "确认选择"}
            </button></div>
          </article>;
        })}</div>
      </>}
  </section>
        {confirmedId && adjustment}
        {focused && <section className="panel itinerary-panel"><div id="plan-detail" className="plan-detail" aria-label={`${styleName(focused.style)}详细行程`}>
          <div className="detail-heading"><div><span className="eyebrow">{showingPreview ? "修改预览 · 尚未保存" : "每日行程"}</span><h3>{styleName(focused.style)}计划</h3></div><span>{focused.days.length} 天 · 步行 {distance(focused.metrics.walking_distance_meters)}</span></div>
          <div className="detail-metrics"><span>兴趣匹配 <strong>{Math.round(focused.metrics.preference_match * 100)}%</strong></span><span>数据置信度 <strong>{Math.round(focused.metrics.data_confidence * 100)}%</strong></span>{focused.score != null && <span>综合排序分 <strong>{focused.score.toFixed(3)}</strong></span>}</div>
          {focused.reason_facts.length > 0 && <div className="plan-reasons"><strong>方案依据</strong><ul>{focused.reason_facts.map((reason, index) => <li key={index}>{reason}</li>)}</ul></div>}
          {focused.validation?.violations.length ? <Issues issues={focused.validation.violations} /> : null}
          {focused.days.map((day, index) => <section className="itinerary-day" key={day.date}>
            <div className="day-heading"><span className="day-number">DAY {String(index + 1).padStart(2, "0")}</span><div><h4>{day.date}</h4><p>{day.theme} · {day.primary_area}</p></div></div>
            {day.start_anchor && <p className="day-anchor">出发 · {day.start_anchor.name}</p>}
            <ol className="itinerary-items">{day.items.map((item, itemIndex) => <li key={item.item_id ?? `${item.name}-${itemIndex}`}>
              <div className="item-time"><strong>{time(item.start_at)}</strong><small>{time(item.end_at)}</small></div>
              <div className="item-content"><span className="item-type">{({ activity: "游览", meal: "用餐", transport: "交通", rest: "休息" } as Record<string, string>)[item.type] ?? "行程"}</span><h5>{item.name}</h5><p>前段交通 {item.travel_from_previous_minutes} 分钟 · {distance(item.distance_from_previous_meters)}<span>预计费用 {money(item.estimated_cost)}</span></p></div>
            </li>)}</ol>
            {day.items.length === 0 && <p className="day-anchor">当天暂无活动安排。</p>}
            {day.end_anchor && <p className="day-anchor end">结束 · {day.end_anchor.name}{day.end_leg && <span> · 末段交通 {day.end_leg.duration_minutes} 分钟{day.end_leg.provider === "mock" ? "（Mock 估算）" : ""}</span>}</p>}
            <div className="day-summary"><span>已知活动费用 {money(day.known_estimated_cost)}</span><span>交通 {day.total_travel_minutes} 分钟</span><span>步行 {distance(day.walking_distance_meters)}</span></div>
          </section>)}
          <p className="cost-note">费用为已知活动费用，未涵盖全部交通、住宿和餐饮。{focused.metrics.unknown_cost_item_count > 0 && `另有 ${focused.metrics.unknown_cost_item_count} 项活动费用待确认。`}</p>
          {focused.assumptions.length > 0 && <div className="plan-reasons"><strong>数据假设与提醒</strong><ul>{focused.assumptions.map((assumption, index) => <li key={index}>{assumption.reason}（{assumption.field}：{assumption.value}）</li>)}</ul></div>}
        </div></section>}
  </>;
}

// 只投影界面需要的领域字段；调试区保留完整原始响应。
export type PlanViolation = {
  type: string;
  severity: string;
  message: string;
  day?: string | null;
  repair_hint?: string | null;
};

export type PlanItem = {
  item_id?: string | null;
  poi_id?: string | null;
  coordinate?: { longitude: number; latitude: number } | null;
  type: string;
  name: string;
  start_at: string;
  end_at: string;
  travel_from_previous_minutes: number;
  distance_from_previous_meters: number;
  estimated_cost: string | number | null;
};

export type PlanDay = {
  date: string;
  theme: string;
  primary_area: string;
  items: PlanItem[];
  known_estimated_cost: string | number;
  unknown_cost_item_count: number;
  total_travel_minutes: number;
  walking_distance_meters: number;
  start_anchor?: { name: string } | null;
  end_anchor?: { name: string } | null;
  end_leg?: { duration_minutes: number; mode: string; provider: string } | null;
};

export type PlanCandidate = {
  id: string;
  style: string;
  days: PlanDay[];
  metrics: {
    known_estimated_cost: string | number;
    unknown_cost_item_count: number;
    total_travel_minutes: number;
    walking_distance_meters: number;
    preference_match: number;
    data_confidence: number;
    fatigue_score: number;
  };
  validation?: { status: string; valid: boolean; violations: PlanViolation[] } | null;
  score?: number | null;
  reason_facts: string[];
  assumptions: { field: string; value: string; reason: string }[];
};

export type PlanningResult = {
  status: string;
  selected_plan: PlanCandidate | null;
  candidates: PlanCandidate[];
  message?: string | null;
};

export type PlanSession = {
  session_id: string;
  status: string;
  session_revision: number;
  candidates: PlanCandidate[];
  allowed_actions: string[];
  active_version: {
    version_id: string;
    number: number;
    selected_candidate_id: string;
    candidate: PlanCandidate;
  } | null;
  pending_preview?: {
    preview_id: string;
    candidate: PlanCandidate;
    status: string;
    hard_validation: NonNullable<PlanCandidate["validation"]>;
    impact: { affected_dates: string[]; reasons: string[] };
    diff: {
      added_items: { name: string; to_date?: string | null }[];
      removed_items: { name: string; from_date?: string | null }[];
      moved_items: { name: string; from_date?: string | null; to_date?: string | null }[];
      reordered_items: { name: string }[];
      time_changes: unknown[];
    };
  } | null;
  message?: string | null;
  interrupt: { id: string; payload: {
    recommended_candidate_id?: string;
    kind?: string;
    approval_token?: string;
    question?: string;
    items?: { item_id: string; name: string; date?: string; day?: string }[];
  } } | null;
};

export type PlanChangeAction =
  | { kind: "edit_text"; text: string }
  | { kind: "approve_preview"; preview_id: string; approval_token: string }
  | { kind: "reject_preview"; preview_id: string }
  | { kind: "clarify_edit"; item_id: string };

export function passesHardValidation(candidate: PlanCandidate): boolean {
  const validation = candidate.validation;
  return !!validation
    && ["valid", "valid_with_warnings"].includes(validation.status)
    && validation.valid !== false
    && !validation.violations.some(issue => issue.severity === "error");
}

// 仅展示当前方案的活动地点；旧响应缺少坐标时不猜测位置。
export function collectPlanPoints(candidate?: PlanCandidate) {
  const seen = new Set<string>();
  return (candidate?.days ?? []).flatMap(day => day.items.flatMap(item => {
    const coordinate = item.coordinate;
    if (item.type !== "activity" || !coordinate
      || !Number.isFinite(coordinate.longitude) || !Number.isFinite(coordinate.latitude)
      || Math.abs(coordinate.longitude) > 180 || Math.abs(coordinate.latitude) > 90) return [];
    const identity = item.poi_id ?? `${coordinate.longitude},${coordinate.latitude}`;
    if (seen.has(identity)) return [];
    seen.add(identity);
    return [{ name: item.name, ...coordinate }];
  }));
}

// The model ladder (#63). Every rung is reported against the same four
// floors and the same oracle gap, so a sophisticated method cannot be graded
// on a friendlier scale than a dumb one.
//
// Two states this table is designed around rather than designed against:
//
//   1. FLOOR-FAIL is a finding, not an error. A trivial baseline out-earning
//      a tuned model is the single most interesting cell in the table, so it
//      gets a legible amber state with the beaten floor NAMED — not a red
//      error box, and not something that only looks right when everything
//      passes.
//   2. A metric can be absent, null-with-a-census (non-finite on every seed),
//      or present but measured over a window too short to interpret it on.
//      All three render distinctly, and none of them renders as zero.
import { AlertTriangle, CheckCircle2, Info } from "lucide-react";
import SplitChip from "./ui/SplitChip.jsx";
import { Tooltip } from "./ui/Tooltip.jsx";
import { cn } from "../lib/cn.js";
import { fmtNum, nonFiniteLabel } from "../lib/format.js";

const FLOORS = [
  ["savings_floor_all_pass", "all_pass"],
  ["savings_floor_random", "random_at_k"],
  ["savings_floor_all_hold", "all_hold"],
  ["savings_floor_volume_rank", "volume_rank"],
];

/**
 * A metric cell's honest state. `windowDays` is the split's own length from
 * the lock's split_boundaries — a time-to-detection longer than the window it
 * was measured on is not a slow detection, it is an unmeasurable one, and
 * saying so beats printing a confident number.
 */
function MetricCell({ row, metric, digits = 3, windowDays }) {
  const value = row.metrics?.[metric];
  const census = row.non_finite?.[metric];

  if (value == null && census) {
    return (
      <Tooltip content={`Non-finite on every seed: ${JSON.stringify(census)}`}>
        <span className="cursor-help text-xs text-notice italic">{nonFiniteLabel(census)}</span>
      </Tooltip>
    );
  }
  if (value == null) {
    if (!(metric in (row.metrics ?? {}))) {
      return (
        <Tooltip content="No seed reported this metric — absent from the artefact, not zero.">
          <span className="cursor-help text-faint italic">not reported</span>
        </Tooltip>
      );
    }
    return <span className="text-faint italic">—</span>;
  }

  const beyondWindow =
    windowDays != null &&
    (metric === "ttd_median_days" ? value > windowDays : false);

  return (
    <span className={cn("tabular font-mono text-xs", beyondWindow ? "text-notice" : "text-muted-foreground")}>
      {fmtNum(value, digits)}
      {beyondWindow && (
        <Tooltip
          content={`Measured on a ${windowDays}-day window. A median TTD beyond the window is not a slow detection — it is one this split cannot resolve.`}
        >
          <span className="ml-[var(--spacing-2)] inline-flex cursor-help items-center gap-[2px] rounded-[var(--radius-2xs)] border border-notice-border px-[4px] text-2xs not-italic">
            <Info aria-hidden="true" className="h-[10px] w-[10px]" />
            beyond window
          </span>
        </Tooltip>
      )}
    </span>
  );
}

function FloorVerdict({ row }) {
  const failed = row.floor_fail ?? [];
  if (row.beats_all_floors) {
    return (
      <span className="inline-flex items-center gap-[var(--spacing-1)] rounded-full border border-positive-border bg-positive-bg px-[var(--spacing-3)] py-[1px] font-mono text-2xs font-bold text-positive uppercase">
        <CheckCircle2 aria-hidden="true" className="h-3 w-3" />
        clears all floors
      </span>
    );
  }
  return (
    <Tooltip
      content={`This rung's savings is below ${failed.join(", ")}. Reported, not hidden: a floor out-earning a trained model is a result about the problem, not a bug in the table.`}
    >
      <span className="inline-flex cursor-help items-center gap-[var(--spacing-1)] rounded-full border border-notice-border bg-notice-bg px-[var(--spacing-3)] py-[1px] font-mono text-2xs font-bold text-notice uppercase">
        <AlertTriangle aria-hidden="true" className="h-3 w-3" />
        under {failed.join(", ") || "a floor"}
      </span>
    </Tooltip>
  );
}

/** Savings against the best floor, as a bar the eye can compare in one pass. */
function SavingsBar({ row }) {
  const savings = row.metrics?.savings;
  const floors = FLOORS.map(([key, name]) => ({ name, value: row.metrics?.[key] })).filter(
    (f) => f.value != null
  );
  if (savings == null || !floors.length) return <span className="text-faint italic">—</span>;
  const best = floors.reduce((a, b) => (a.value > b.value ? a : b));
  const scale = Math.max(Math.abs(savings), Math.abs(best.value), 1e-9);
  const pct = (v) => `${Math.max(2, (Math.abs(v) / scale) * 100)}%`;
  const wins = savings >= best.value;

  return (
    <div className="min-w-[132px]">
      <div className="flex items-center gap-[var(--spacing-2)]">
        <span
          className={cn("h-[6px] rounded-full", wins ? "bg-positive" : "bg-primary")}
          style={{ width: pct(savings) }}
        />
        <span className="tabular font-mono text-2xs text-foreground">{fmtNum(savings, 3)}</span>
      </div>
      <div className="mt-[3px] flex items-center gap-[var(--spacing-2)]">
        <span
          className={cn("h-[6px] rounded-full", wins ? "bg-faint" : "bg-notice")}
          style={{ width: pct(best.value) }}
        />
        <span className="tabular font-mono text-2xs text-faint">
          {fmtNum(best.value, 3)} · {best.name}
        </span>
      </div>
    </div>
  );
}

export default function LadderTable({ ladder, windowDaysBySplit = {} }) {
  const rungs = ladder?.payload?.rungs ?? [];

  return (
    <div className="overflow-x-auto rounded-[var(--radius-sm)] border border-border">
      <table className="w-full min-w-[1080px] border-collapse text-sm">
        <thead>
          <tr>
            {[
              "rung",
              "split",
              "seeds",
              "PR-AUC",
              "savings vs best floor",
              "floor verdict",
              "gap to oracle",
              "P@K",
              "R@K",
              "TTD (days)",
              "d30 detection",
              "ECE",
            ].map((h) => (
              <th
                key={h}
                className="sticky top-0 z-[1] border-b border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-3)] text-left text-2xs font-semibold tracking-wide text-faint uppercase"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rungs.map((row) => {
            const windowDays = windowDaysBySplit[row.split];
            return (
              <tr
                key={`${row.rung}-${row.label}-${row.split}-${row.cost_scenario}`}
                className={cn(
                  "border-b border-border last:border-0 hover:bg-canvas-well",
                  row.beats_all_floors
                    ? "shadow-[inset_3px_0_0_var(--color-positive)]"
                    : "shadow-[inset_3px_0_0_var(--color-notice)]"
                )}
              >
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <div className="flex items-center gap-[var(--spacing-2)]">
                    <span className="tabular font-mono text-2xs text-faint">R{row.rung}</span>
                    <span className="font-mono text-xs font-semibold text-foreground">{row.label}</span>
                  </div>
                  {!row.provenance_consistent && (
                    <Tooltip content={`Rows in this group disagree on harness or commit: ${row.eval_lock_sha?.join(", ")}`}>
                      <span className="mt-[2px] inline-block cursor-help font-mono text-2xs text-negative">
                        mixed provenance
                      </span>
                    </Tooltip>
                  )}
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <SplitChip split={row.split} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <Tooltip content={`seeds: ${(row.seeds ?? []).join(", ") || "unlabelled"}`}>
                    <span className="tabular cursor-help font-mono text-xs text-muted-foreground">
                      {row.n_seeds}
                    </span>
                  </Tooltip>
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="pr_auc" digits={4} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <SavingsBar row={row} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <FloorVerdict row={row} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="gap_to_oracle" digits={3} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="precision_at_k" digits={3} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="recall_at_k" digits={3} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="ttd_median_days" digits={1} windowDays={windowDays} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="detection_rate_d30" digits={3} />
                </td>
                <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                  <MetricCell row={row} metric="ece" digits={4} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// The lock artefact, rendered rather than claimed in prose (#63's AC): the
// three hashes, the open counter, and the commit that wrote EVAL-LOCK.json,
// for every lock in the supersession chain — not only the authoritative one,
// so a reader can see cycle 1 was superseded rather than simply vanishing.
import { GitCommitHorizontal, ShieldCheck } from "lucide-react";
import { Tooltip } from "./ui/Tooltip.jsx";
import { cn } from "../lib/cn.js";

function Hash({ label, value }) {
  return (
    <div>
      <dt className="m-0 text-2xs font-semibold tracking-wide text-faint uppercase">{label}</dt>
      <Tooltip content={<span className="font-mono break-all">{value}</span>}>
        <dd className="m-0 mt-[2px] cursor-help font-mono text-xs text-primary-text">
          {value ? `${value.slice(0, 16)}…` : "—"}
        </dd>
      </Tooltip>
    </div>
  );
}

export default function LockPanel({ lockState, variant = "full" }) {
  const locks = lockState?.payload?.locks ?? [];
  const authoritative = lockState?.payload?.authoritative_lock;
  const testOpen = lockState?.payload?.test_split_opened;

  return (
    <div className="grid gap-[var(--spacing-4)]">
      <div className="flex flex-wrap items-center gap-[var(--spacing-3)] text-sm text-muted-foreground">
        <ShieldCheck className="h-4 w-4 text-primary" aria-hidden="true" />
        <span>
          <strong className="font-semibold text-foreground">{locks.length}</strong> lock
          {locks.length === 1 ? "" : "s"} in the supersession chain — authoritative:{" "}
          <span className="font-mono text-primary-text">{authoritative ?? "—"}</span>
        </span>
        <span
          className={cn(
            "ml-auto rounded-full border px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold uppercase",
            testOpen ? "border-positive-border bg-positive-bg text-positive" : "border-notice-border bg-notice-bg text-notice"
          )}
        >
          test split {testOpen ? "opened" : "shut"}
        </span>
      </div>

      {locks.map((lock) => (
        <div
          key={lock.file}
          className={cn(
            "rounded-[var(--radius-md)] border p-[var(--spacing-5)]",
            lock.authoritative ? "border-primary/40 bg-canvas-well" : "border-border"
          )}
        >
          <div className="flex flex-wrap items-center justify-between gap-[var(--spacing-3)]">
            <div className="flex items-center gap-[var(--spacing-2)]">
              <span className="font-mono text-sm font-semibold text-foreground">{lock.file}</span>
              <span className="rounded-full border border-border px-[var(--spacing-2)] py-[1px] font-mono text-2xs text-faint">
                cycle {lock.cycle}
              </span>
              {lock.authoritative && (
                <span className="rounded-full border border-primary/40 bg-primary/10 px-[var(--spacing-2)] py-[1px] font-mono text-2xs font-bold text-primary uppercase">
                  authoritative
                </span>
              )}
              {lock.superseded_by && (
                <span className="font-mono text-2xs text-faint">superseded by {lock.superseded_by}</span>
              )}
            </div>
            <div className="flex items-center gap-[var(--spacing-2)] text-xs text-faint">
              <GitCommitHorizontal className="h-3.5 w-3.5" aria-hidden="true" />
              <Tooltip content={<span className="font-mono break-all">{lock.frozen_at_git_sha}</span>}>
                <span className="cursor-help font-mono">
                  {lock.frozen_at_git_sha ? `${lock.frozen_at_git_sha.slice(0, 10)}…` : "—"}
                </span>
              </Tooltip>
            </div>
          </div>

          {variant === "full" && (
            <>
              <dl className="mt-[var(--spacing-4)] grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-[var(--spacing-4)]">
                <Hash label="eval_module_sha256" value={lock.hashes?.eval_module_sha256} />
                <Hash label="generator_module_sha256" value={lock.hashes?.generator_module_sha256} />
                <Hash label="scenario_config_sha256" value={lock.hashes?.scenario_config_sha256} />
              </dl>
              <div className="mt-[var(--spacing-4)] flex flex-wrap gap-[var(--spacing-5)] text-xs text-muted-foreground">
                <span>
                  open counter: <strong className="text-foreground">{lock.open_count}</strong>
                </span>
                <span>
                  seeds: <strong className="font-mono text-foreground">{(lock.seeds ?? []).join(", ") || "—"}</strong>
                </span>
                <span>
                  capacity_k: <strong className="font-mono text-foreground">{lock.capacity_k ?? "—"}</strong>
                </span>
                {lock.pre_registration && (
                  <span>
                    pre-registration: <span className="font-mono text-primary-text">{lock.pre_registration}</span>
                  </span>
                )}
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

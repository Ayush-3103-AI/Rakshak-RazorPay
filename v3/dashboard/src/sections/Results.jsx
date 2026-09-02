// §3 — the payoff, in three figures (#79).
//
// Absorbs what TrajectoryLadder.jsx used to render, minus two things that moved:
// its hand-written v1/v2 waypoints are now §1, built from journey.json, and its
// lock panel is now §2, where a reader meets it BEFORE a number rather than after.
// The ladder itself is unchanged.
//
// Two figures are new. The cost sweep had run once, over the whole ladder, and
// existed only as a markdown table — which meant the project's most robust result
// was the one nobody could see. The HOLD decomposition sits directly beneath it
// because the honest reading of the sweep is incomplete without it: a large part
// of the margin is the decision layer rather than the ranker, and burying that
// would be the exact move this panel exists to refuse.
import { motion, useReducedMotion } from "framer-motion";
import { Scale, Table2, TrendingUp } from "lucide-react";
import CostSweepChart from "../components/CostSweepChart.jsx";
import LadderTable from "../components/LadderTable.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum, fmtSigned } from "../lib/format.js";
import G5Figure from "./G5Figure.jsx";

// Which arm of the sweep to plot by default. `realised` is the arm the adopted
// row is scored under; the others are reachable in cost_sweep.json and named in
// the artifact's own `arm_note`, which is printed below the chart rather than
// paraphrased here.
const DEFAULT_ARM = "realised";

export default function Results() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");
  const sweep = useArtifact("cost_sweep");
  const reduce = useReducedMotion();

  const locks = lockState.data?.payload?.locks ?? [];
  const live = locks.find((l) => l.authoritative);
  const provenance = ladder.data?.provenance;

  // Split length from the lock itself, so "beyond the window" is the lock's
  // arithmetic rather than this component's opinion.
  const windowDaysBySplit = {};
  for (const [name, bounds] of Object.entries(live?.split_boundaries ?? {})) {
    if (Array.isArray(bounds) && bounds.length === 2) {
      const key = { train: "TRAIN", val: "VALIDATION", test: "TEST" }[name] ?? name.toUpperCase();
      windowDaysBySplit[key] = bounds[1] - bounds[0] + 1;
    }
  }

  const sweepPayload = sweep.data?.payload;
  const arm = sweepPayload?.arms?.[DEFAULT_ARM] ?? [];
  const decomposition = sweepPayload?.hold_decomposition ?? [];

  return (
    <div>
      {/* ---- §3a the ladder ---------------------------------------------- */}
      <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
        <div className="mx-auto max-w-[1180px]">
          <p className="m-0 mb-[var(--spacing-3)] flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
            <Table2 aria-hidden="true" className="h-4 w-4" />
            §3a · The ladder
          </p>
          <h2 className="m-0 max-w-[30ch] font-heading text-3xl font-bold tracking-tight text-foreground">
            Every policy, against the same floors
          </h2>
          <p className="mt-[var(--spacing-4)] max-w-[74ch] text-base leading-relaxed text-muted-foreground">
            Every rung against the same floors and the same oracle gap. A rung that loses to a floor is
            marked as losing to <em>that named floor</em> — the comparison is the point, and a trivial
            baseline out-earning a trained model is a result about the problem rather than a fault in
            the table.
          </p>

          <div className="mt-[var(--spacing-5)]">
            {ladder.loading && <ArtifactLoading label="Loading ladder.json…" />}
            {ladder.error && <ArtifactError artifact="ladder" error={ladder.error} />}
            {ladder.data && (
              <>
                <LadderTable ladder={ladder.data} windowDaysBySplit={windowDaysBySplit} />

                {provenance && (
                  <Card pad="compact" elevation="low" className="mt-[var(--spacing-4)]">
                    <p className="m-0 text-xs leading-relaxed text-muted-foreground">
                      <strong className="text-foreground">Provenance.</strong>{" "}
                      {provenance.results_are_current === false ? (
                        <>
                          At least one row was scored under a{" "}
                          <span className="font-semibold text-notice">superseded cycle</span> and is
                          rendered as such — that is the pre-registered state for a rung the newer cycle
                          did not rescore, not drift.
                        </>
                      ) : (
                        <>Every row was scored under the authoritative lock.</>
                      )}{" "}
                      {Object.entries(provenance.results_scored_under ?? {}).map(([sha, entry]) => (
                        <span key={sha} className="mr-[var(--spacing-3)] font-mono text-2xs">
                          {sha.slice(0, 8)}… → cycle {entry.cycles?.join("/")} ({entry.sources?.length} rows)
                        </span>
                      ))}
                    </p>
                    <p className="m-0 mt-[var(--spacing-3)] text-2xs leading-relaxed text-faint">
                      {provenance.harness_note}
                    </p>
                  </Card>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* ---- §3b the cost sweep ------------------------------------------ */}
      <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
        <div className="mx-auto max-w-[1180px]">
          <p className="m-0 mb-[var(--spacing-3)] flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
            <TrendingUp aria-hidden="true" className="h-4 w-4" />
            §3b · Cost-asymmetry sweep
          </p>
          <h2 className="m-0 max-w-[32ch] font-heading text-3xl font-bold tracking-tight text-foreground">
            Does the ranking survive a different cost matrix?
          </h2>
          <p className="mt-[var(--spacing-4)] max-w-[74ch] text-base leading-relaxed text-muted-foreground">
            Every savings number this project published before this sweep ran was a{" "}
            <strong className="text-foreground">single point estimate at one cost matrix</strong>. The
            sweep re-prices the decisions the committed models already make across four orders of
            magnitude of false-hold-to-fraud-loss asymmetry. Nothing was refitted; only the price of a
            wrong answer changed.
          </p>

          {sweep.loading && <ArtifactLoading label="Loading cost_sweep.json…" />}
          {sweep.error && <ArtifactError artifact="cost_sweep" error={sweep.error} />}

          {sweepPayload && (
            <>
              <div className="mt-[var(--spacing-5)] flex flex-wrap items-center gap-[var(--spacing-3)]">
                <SplitChip split={sweep.data?.split} />
                <span className="font-mono text-2xs text-faint">
                  arm: {DEFAULT_ARM} · K = {fmtNum(sweepPayload.meta?.k, 0)} ·{" "}
                  {fmtNum(sweepPayload.meta?.n_merchants, 0)} merchants ·{" "}
                  {sweepPayload.meta?.seeds?.length ?? 0} seeds
                </span>
                {sweepPayload.shipped_ratio != null && (
                  <span
                    className={
                      sweepPayload.shipped_ratio_within_grid
                        ? "rounded-full border border-positive-border bg-positive-bg px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold text-positive uppercase"
                        : "rounded-full border border-notice-border bg-notice-bg px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold text-notice uppercase"
                    }
                  >
                    shipped ratio {fmtNum(sweepPayload.shipped_ratio, 5)} —{" "}
                    {sweepPayload.shipped_ratio_within_grid ? "inside the grid" : "OUTSIDE the grid"}
                  </span>
                )}
              </div>

              <div className="mt-[var(--spacing-5)] overflow-x-auto">
                <CostSweepChart
                  ratios={sweepPayload.ratios}
                  series={arm}
                  shippedRatio={sweepPayload.shipped_ratio}
                  shippedWithinGrid={sweepPayload.shipped_ratio_within_grid}
                  animate={!reduce}
                />
              </div>

              <p className="mt-[var(--spacing-4)] max-w-[86ch] text-xs leading-relaxed text-faint">
                {sweepPayload.arm_note}
              </p>

              {/* ---- the decomposition, printed rather than buried ---- */}
              {decomposition.length > 0 && (
                <div className="mt-[var(--spacing-8)]">
                  <h3 className="m-0 mb-[var(--spacing-2)] flex items-center gap-[var(--spacing-2)] font-heading text-xl font-bold text-foreground">
                    <Scale aria-hidden="true" className="h-5 w-5 text-primary" />
                    Where the margin actually comes from
                  </h3>
                  <p className="mt-0 mb-[var(--spacing-4)] max-w-[74ch] text-sm text-muted-foreground">
                    The same rows, the same selector, the same top-K — with the HOLD action made
                    unreachable and nothing else changed. What the bar loses is what the{" "}
                    <em>decision layer</em> was worth, as distinct from the ranking.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[520px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-border-strong text-left">
                          <th className="py-[var(--spacing-3)] pr-[var(--spacing-4)] font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                            policy
                          </th>
                          <th className="py-[var(--spacing-3)] pr-[var(--spacing-4)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                            with HOLD
                          </th>
                          <th className="py-[var(--spacing-3)] pr-[var(--spacing-4)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                            HOLD forbidden
                          </th>
                          <th className="py-[var(--spacing-3)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                            what HOLD is worth
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {decomposition.map((row, i) => (
                          <motion.tr
                            key={row.policy}
                            initial={reduce ? false : { opacity: 0, x: -8 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true, margin: "-30px" }}
                            transition={{ delay: reduce ? 0 : i * 0.05, duration: 0.35, ease: [0, 0, 0.2, 1] }}
                            className="border-b border-border"
                          >
                            <td className="py-[var(--spacing-3)] pr-[var(--spacing-4)] font-mono text-foreground">
                              {row.policy}
                            </td>
                            <td className="py-[var(--spacing-3)] pr-[var(--spacing-4)] text-right font-mono tabular-nums text-muted-foreground">
                              {fmtNum(row.with_hold, 4)}
                            </td>
                            <td className="py-[var(--spacing-3)] pr-[var(--spacing-4)] text-right font-mono tabular-nums text-muted-foreground">
                              {fmtNum(row.without_hold, 4)}
                            </td>
                            <td className="py-[var(--spacing-3)] text-right font-mono font-semibold tabular-nums text-foreground">
                              {fmtSigned(row.delta, 4)}
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-[var(--spacing-4)] max-w-[80ch] text-xs leading-relaxed text-faint">
                    Taken at swept ratio {fmtNum(sweepPayload.hold_decomposition_at_ratio, 5)}
                    {sweepPayload.hold_decomposition_anchored
                      ? " — the point on the grid nearest the shipped cost matrix"
                      : " — the grid's midpoint, because the sweep reported no usable shipped ratio to anchor to"}
                    . Read this as a limit on the claim, not
                    a footnote to it: the advantage is a decision-layer result, and priced as a raw
                    REVIEW-only ranking the picture changes — that arm is in the same artifact.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* ---- §3c the confounder null ------------------------------------- */}
      <G5Figure eyebrow="§3c · G5 confounder null" />
    </div>
  );
}

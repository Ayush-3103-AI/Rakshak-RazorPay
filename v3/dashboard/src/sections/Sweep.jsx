// §3 — the cost-asymmetry sweep, beside the decomposition that qualifies it.
//
// These two belong on one screen and nowhere near each other in importance. The
// sweep is the project's most robust result: every savings figure published
// before it ran was a single point estimate at one cost matrix. The
// decomposition is the limit on that result — roughly half the margin is the
// decision layer rather than the ranker.
//
// So they sit side by side, at the same visual weight, on purpose. Putting the
// decomposition below the fold of a scrolling section is how a caveat becomes a
// footnote, and burying it is the exact move this panel exists to refuse.
import { motion, useReducedMotion } from "framer-motion";
import CostSweepChart from "../components/CostSweepChart.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum, fmtSigned } from "../lib/format.js";

// Which arm of the sweep to plot by default. `realised` is the arm the adopted
// row is scored under; the others are reachable in cost_sweep.json and named in
// the artifact's own `arm_note`, which is printed below the chart rather than
// paraphrased here.
const DEFAULT_ARM = "realised";

export default function Sweep() {
  const sweep = useArtifact("cost_sweep");
  const reduce = useReducedMotion();

  const payload = sweep.data?.payload;
  const arm = payload?.arms?.[DEFAULT_ARM] ?? [];
  const decomposition = payload?.hold_decomposition ?? [];

  return (
    <Page
      id="sweep"
      eyebrow="§3 · Evidence"
      title="It still wins when you change the price of being wrong."
      lede={
        <>
          The same committed decisions, re-priced across four orders of magnitude of
          false-hold-to-fraud-loss asymmetry. Nothing was refitted — only the cost of a wrong answer
          changed.
        </>
      }
      actions={
        <>
          <SplitChip split={sweep.data?.split} />
          {payload?.shipped_ratio != null && (
            <span
              className={
                payload.shipped_ratio_within_grid
                  ? "rounded-full border border-positive-border bg-positive-bg px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold text-positive uppercase"
                  : "rounded-full border border-notice-border bg-notice-bg px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold text-notice uppercase"
              }
            >
              shipped ratio {fmtNum(payload.shipped_ratio, 5)} —{" "}
              {payload.shipped_ratio_within_grid ? "inside the grid" : "OUTSIDE the grid"}
            </span>
          )}
        </>
      }
    >
      {sweep.loading && <ArtifactLoading label="Loading cost_sweep.json…" />}
      {sweep.error && <ArtifactError artifact="cost_sweep" error={sweep.error} />}

      {payload && (
        <div className="grid h-full min-h-0 grid-cols-[1.5fr_1fr] gap-[var(--spacing-4)] max-xl:grid-cols-1">
          <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-[var(--spacing-3)]">
              <h3 className="m-0 font-heading text-base font-semibold text-foreground">
                Savings across the swept cost matrix
              </h3>
              <span className="font-mono text-2xs text-faint">
                arm: {DEFAULT_ARM} · K = {fmtNum(payload.meta?.k, 0)} ·{" "}
                {fmtNum(payload.meta?.n_merchants, 0)} merchants · {payload.meta?.seeds?.length ?? 0} seeds
              </span>
            </div>
            <div className="mt-[var(--spacing-4)] min-h-0 flex-1 overflow-auto">
              <CostSweepChart
                ratios={payload.ratios}
                series={arm}
                shippedRatio={payload.shipped_ratio}
                shippedWithinGrid={payload.shipped_ratio_within_grid}
                animate={!reduce}
              />
            </div>
            <p className="m-0 mt-[var(--spacing-3)] shrink-0 text-2xs leading-relaxed text-faint">
              {payload.arm_note}
            </p>
          </Card>

          {decomposition.length > 0 && (
            <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
              <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">
                Where the margin actually comes from
              </h3>
              <p className="m-0 mt-[var(--spacing-2)] shrink-0 text-xs leading-relaxed text-muted-foreground">
                Same rows, same selector, same top-K — with the HOLD action made unreachable and nothing
                else changed. What the number loses is what the <em>decision layer</em> was worth, as
                distinct from the ranking.
              </p>

              <div className="mt-[var(--spacing-4)] min-h-0 flex-1 overflow-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-border-strong text-left">
                      <th className="sticky top-0 bg-card py-[var(--spacing-3)] pr-[var(--spacing-3)] font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                        policy
                      </th>
                      <th className="sticky top-0 bg-card py-[var(--spacing-3)] pr-[var(--spacing-3)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                        with HOLD
                      </th>
                      <th className="sticky top-0 bg-card py-[var(--spacing-3)] pr-[var(--spacing-3)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                        HOLD forbidden
                      </th>
                      <th className="sticky top-0 bg-card py-[var(--spacing-3)] text-right font-mono text-2xs font-bold tracking-wider text-faint uppercase">
                        worth
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {decomposition.map((row, i) => (
                      <motion.tr
                        key={row.policy}
                        initial={reduce ? false : { opacity: 0, x: -8 }}
                        whileInView={{ opacity: 1, x: 0 }}
                        viewport={{ once: true, margin: "-20px" }}
                        transition={{ delay: reduce ? 0 : i * 0.05, duration: 0.35, ease: [0, 0, 0.2, 1] }}
                        className="border-b border-border last:border-0"
                      >
                        <td className="py-[var(--spacing-3)] pr-[var(--spacing-3)] font-mono text-foreground">
                          {row.policy}
                        </td>
                        <td className="py-[var(--spacing-3)] pr-[var(--spacing-3)] text-right font-mono tabular-nums text-muted-foreground">
                          {fmtNum(row.with_hold, 4)}
                        </td>
                        <td className="py-[var(--spacing-3)] pr-[var(--spacing-3)] text-right font-mono tabular-nums text-muted-foreground">
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

              <p className="m-0 mt-[var(--spacing-3)] shrink-0 border-t border-border pt-[var(--spacing-3)] text-2xs leading-relaxed text-faint">
                At swept ratio {fmtNum(payload.hold_decomposition_at_ratio, 5)}
                {payload.hold_decomposition_anchored
                  ? " — the grid point nearest the shipped cost matrix"
                  : " — the grid's midpoint, no usable shipped ratio to anchor to"}
                . Read this as a limit on the claim, not a footnote to it.
              </p>
            </Card>
          )}
        </div>
      )}
    </Page>
  );
}

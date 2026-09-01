// #63 — the research trajectory, then the ladder measured against it.
//
// The v1/v2 waypoints below are the only literals on this panel, and they are
// literals on purpose: the v1 harness is closed forever (00-charter-v2.md §6.1
// — "No v1 number is recomputed, corrected, or improved"), so those two
// figures are historical text, not artefact data, and no generator will ever
// re-emit them. Every v3 number in this section comes from ladder.json and
// lock_state.json.
import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, ScrollText } from "lucide-react";
import LadderTable from "../components/LadderTable.jsx";
import LockPanel from "../components/LockPanel.jsx";
import Card from "../components/ui/Card.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { cn } from "../lib/cn.js";

const WAYPOINTS = [
  {
    era: "v1",
    tone: "negative",
    title: "The HMM hypothesis was falsified",
    body: "LightGBM beat v1 by 0.3176 PR-AUC. The bar moved to LightGBM and stayed there — moving it back to the rule engine would have been the dishonest repair.",
    citation: "project-context/00-charter-v2.md §2",
  },
  {
    era: "v1",
    tone: "negative",
    title: "K2 missed at 5.9% against a 20% bar",
    body: "The v1 harness is closed forever: that miss stands in the retrospective exactly as measured. No v1 number is recomputed, corrected or improved.",
    citation: "project-context/00-charter-v2.md §6.1",
  },
  {
    era: "v2",
    tone: "neutral",
    title: "A new harness, hashed before any model existed",
    body: "v2 was re-frozen into its own lock with the open counter committed, so the test split can be opened exactly once and the fact of it is checkable rather than asserted.",
    citation: "project-context/00-charter-v2.md §6.2–6.3",
  },
];

const TONE = {
  negative: "border-negative-border bg-negative-bg",
  neutral: "border-information-border bg-information-bg",
  positive: "border-positive-border bg-positive-bg",
};

export default function TrajectoryLadder() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");
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

  const rungCount = ladder.data?.payload?.rungs?.length ?? 0;

  return (
    <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          §2 · Trajectory &amp; ladder
        </p>
        <h2 className="m-0 max-w-[28ch] font-heading text-3xl font-bold tracking-tight text-foreground">
          Three cycles, one harness discipline
        </h2>
        <p className="mt-[var(--spacing-4)] max-w-[68ch] text-base leading-relaxed text-muted-foreground">
          A single clean number would be a weaker claim than the arc that produced it. v1's falsification
          and v2's re-freeze are reported beside v3's ladder, unmodified, because "the hypothesis was wrong
          and here is the diagnosis" is both stronger and true.
        </p>

        <div className="mt-[var(--spacing-7)] grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-[var(--spacing-4)]">
          {WAYPOINTS.map((w, i) => (
            <motion.div
              key={w.title}
              initial={reduce ? false : { opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: reduce ? 0 : i * 0.07, duration: 0.45, ease: [0, 0, 0.2, 1] }}
              className={cn("rounded-[var(--radius-md)] border border-l-4 p-[var(--spacing-5)]", TONE[w.tone])}
            >
              <span className="font-mono text-2xs font-bold tracking-widest text-faint uppercase">{w.era}</span>
              <h3 className="m-0 mt-[var(--spacing-2)] font-heading text-base font-semibold text-foreground">
                {w.title}
              </h3>
              <p className="m-0 mt-[var(--spacing-2)] text-sm leading-relaxed text-muted-foreground">{w.body}</p>
              <p className="m-0 mt-[var(--spacing-3)] flex items-center gap-[var(--spacing-2)] font-mono text-2xs text-faint">
                <ScrollText aria-hidden="true" className="h-3 w-3 shrink-0" />
                {w.citation}
              </p>
            </motion.div>
          ))}

          <motion.div
            initial={reduce ? false : { opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-40px" }}
            transition={{ delay: reduce ? 0 : 0.21, duration: 0.45, ease: [0, 0, 0.2, 1] }}
            className="rounded-[var(--radius-md)] border border-l-4 border-primary/40 bg-canvas-well p-[var(--spacing-5)]"
          >
            <span className="font-mono text-2xs font-bold tracking-widest text-primary-text uppercase">v3 · live</span>
            <h3 className="m-0 mt-[var(--spacing-2)] flex items-center gap-[var(--spacing-2)] font-heading text-base font-semibold text-foreground">
              Cycle {live?.cycle ?? "—"} <ArrowRight aria-hidden="true" className="h-4 w-4 text-faint" />{" "}
              <span className="tabular font-mono">{rungCount}</span> scored rows
            </h3>
            <p className="m-0 mt-[var(--spacing-2)] text-sm leading-relaxed text-muted-foreground">
              Read live from <code className="font-mono text-xs">ladder.json</code> and{" "}
              <code className="font-mono text-xs">lock_state.json</code> — this card has no numbers of its own,
              which is why it cannot drift from the artefacts.
            </p>
          </motion.div>
        </div>

        {/* ---- the lock, displayed rather than claimed ---- */}
        <div className="mt-[var(--spacing-9)]">
          <h3 className="m-0 mb-[var(--spacing-2)] font-heading text-xl font-bold text-foreground">
            The lock these numbers were scored under
          </h3>
          <p className="mt-0 mb-[var(--spacing-4)] max-w-[70ch] text-sm text-muted-foreground">
            Three hashes, the open counter and the freezing commit, for every lock in the chain.
          </p>
          {lockState.loading && <ArtifactLoading label="Loading lock_state.json…" />}
          {lockState.error && <ArtifactError artifact="lock_state" error={lockState.error} />}
          {lockState.data && <LockPanel lockState={lockState.data} variant="full" />}
        </div>

        {/* ---- the ladder ---- */}
        <div className="mt-[var(--spacing-9)]">
          <h3 className="m-0 mb-[var(--spacing-2)] font-heading text-xl font-bold text-foreground">The model ladder</h3>
          <p className="mt-0 mb-[var(--spacing-4)] max-w-[74ch] text-sm text-muted-foreground">
            Every rung against the same four floors and the same oracle gap. A rung that loses to a floor is
            marked as losing to <em>that named floor</em> — the comparison is the point, and a trivial
            baseline out-earning a trained model is a result about the problem rather than a fault in the
            table.
          </p>

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
                        <span className="font-semibold text-notice">superseded cycle</span> and is rendered as
                        such — that is the pre-registered state for a rung the newer cycle did not rescore, not
                        drift.
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
  );
}

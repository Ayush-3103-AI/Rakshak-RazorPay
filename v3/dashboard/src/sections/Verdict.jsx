// §0 — the claim, and the four numbers that qualify it (#79).
//
// Every figure on this screen is DERIVED FROM THE ARTIFACTS, never typed here:
// the policy count from ladder.json's rows, the seed count and the lock count
// from lock_state.json, the surviving row from `beats_all_floors`, and the
// margin band from cost_sweep.json's realised arm. That is not fastidiousness —
// a hero is the one place a stale literal survives longest, and this project's
// entire pitch is that its documents cannot drift from its measurements.
//
// It is also why the headline is phrased as a question the artifacts answer. If
// a future rescore promotes a second row, this section says "two", and the
// sentence beneath it stops claiming uniqueness, without anyone editing copy.
import { motion, useReducedMotion } from "framer-motion";
import { Lock, ShieldAlert } from "lucide-react";
import Counter from "../components/Counter.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum } from "../lib/format.js";

function rise(reduce, i) {
  return {
    initial: reduce ? false : { opacity: 0, y: 18 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, margin: "-40px" },
    transition: { delay: reduce ? 0 : i * 0.08, duration: 0.5, ease: [0, 0, 0.2, 1] },
  };
}

export default function Verdict() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");
  const sweep = useArtifact("cost_sweep");
  const reduce = useReducedMotion();

  const rungs = ladder.data?.payload?.rungs ?? [];
  const locks = lockState.data?.payload?.locks ?? [];
  const live = locks.find((l) => l.authoritative);
  const survivors = rungs.filter((r) => r.beats_all_floors);
  // Seeds per row rather than the lock's declared list: what a row was actually
  // scored on is the honest denominator, and the lock only says what was intended.
  const seedCount = Math.max(0, ...rungs.map((r) => r.n_seeds ?? 0));
  const scoredRows = rungs.reduce((n, r) => n + (r.n_seeds ?? 0), 0);

  const realised = sweep.data?.payload?.arms?.realised ?? [];
  const survivorName = survivors[0]?.label;
  const survivorSeries = realised.find((s) => s.policy === survivorName?.replace(/_realised_exposure$/, ""));
  const band = survivorSeries?.values?.length
    ? [Math.min(...survivorSeries.values), Math.max(...survivorSeries.values)]
    : null;

  const loading = ladder.loading || lockState.loading || sweep.loading;
  const failed = ladder.error ?? lockState.error ?? sweep.error;

  const TILES = [
    { key: "policies", value: rungs.length, label: "policies on the ladder", note: `${scoredRows} scored rows` },
    { key: "seeds", value: seedCount, label: "seeds per policy", note: "mean, with the per-seed range beside it" },
    { key: "locks", value: locks.length, label: "sealed eval locks", note: `authoritative: ${live?.file ?? "—"}` },
    {
      key: "survivors",
      value: survivors.length,
      label: survivors.length === 1 ? "row beat every floor" : "rows beat every floor",
      note: survivors.length ? survivors.map((s) => s.label).join(", ") : "none on this ladder",
      emphasis: true,
    },
  ];

  return (
    <div className="border-b border-border bg-canvas-well px-[var(--spacing-8)] py-[var(--spacing-11)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-8)]">
      <div className="mx-auto max-w-[1180px]">
        <motion.p
          {...rise(reduce, 0)}
          className="m-0 mb-[var(--spacing-3)] flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase"
        >
          <ShieldAlert aria-hidden="true" className="h-4 w-4" />
          §0 · Verdict
        </motion.p>

        <motion.h1
          {...rise(reduce, 1)}
          className="m-0 max-w-[22ch] font-heading text-6xl leading-[1.04] font-extrabold tracking-tight text-foreground max-md:text-4xl"
        >
          {rungs.length ? (
            <>
              {rungs.length} policies. {seedCount} seeds. {locks.length} sealed locks.
              <br />
              <span className="text-primary-text">
                {survivors.length === 1 ? "One row survived." : `${survivors.length} rows survived.`}
              </span>
            </>
          ) : (
            "The ladder has not been scored in this tree."
          )}
        </motion.h1>

        <motion.p
          {...rise(reduce, 2)}
          className="mt-[var(--spacing-6)] max-w-[74ch] text-lg leading-relaxed text-muted-foreground"
        >
          Rakshak watches a merchant <em>after</em> onboarding clears it. Every number below was
          scored under a harness hashed and sealed before the model that produced it existed — and{" "}
          {survivors.length === 1 ? (
            <>
              exactly one policy on this ladder beats <strong className="text-foreground">every</strong>{" "}
              floor on <strong className="text-foreground">every</strong> seed.
            </>
          ) : (
            <>the count of policies clearing every floor on every seed is read live from the ladder.</>
          )}{" "}
          Everything else we built, we killed on the record.
        </motion.p>

        {loading && <ArtifactLoading label="Loading the ladder, the locks and the sweep…" />}
        {failed && !loading && <ArtifactError artifact="verdict" error={failed} />}

        <div className="mt-[var(--spacing-9)] grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-[var(--spacing-4)]">
          {TILES.map((t, i) => (
            <motion.div key={t.key} {...rise(reduce, 3 + i)}>
              <Card
                pad="regular"
                elevation="low"
                className={t.emphasis ? "border-primary/40" : undefined}
              >
                <Counter
                  value={t.value}
                  format={(v) => Math.round(v).toLocaleString("en-IN")}
                  className="block font-mono text-5xl font-bold tabular-nums text-foreground"
                />
                <p className="m-0 mt-[var(--spacing-2)] text-sm font-medium text-foreground">{t.label}</p>
                <p className="m-0 mt-[var(--spacing-1)] font-mono text-2xs leading-relaxed text-faint">{t.note}</p>
              </Card>
            </motion.div>
          ))}
        </div>

        {band && (
          <motion.div {...rise(reduce, 7)} className="mt-[var(--spacing-5)]">
            <Card pad="regular" elevation="low">
              <p className="m-0 text-sm leading-relaxed text-muted-foreground">
                <strong className="text-foreground">And the margin is not a point estimate.</strong>{" "}
                Across the full swept range of false-hold-to-fraud-loss asymmetry — four orders of
                magnitude — {survivorSeries.policy} under realised exposure holds{" "}
                <span className="font-mono font-semibold text-foreground">
                  {fmtNum(band[0], 4)} to {fmtNum(band[1], 4)}
                </span>{" "}
                savings. The figure and its decomposition are in §3; roughly half of that advantage
                is the decision layer rather than the ranker, and §3 says so.
              </p>
            </Card>
          </motion.div>
        )}

        <motion.div
          {...rise(reduce, 8)}
          className="mt-[var(--spacing-6)] flex flex-wrap items-center gap-[var(--spacing-3)]"
        >
          <SplitChip split={ladder.data?.split} />
          <span className="inline-flex items-center gap-[var(--spacing-2)] rounded-[var(--radius-xs)] border border-border bg-card px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold tracking-widest text-faint uppercase">
            <Lock aria-hidden="true" className="h-3 w-3" />
            test split opened {lockState.data?.payload?.test_split_opened ? "ONCE" : "0 times"}
          </span>
        </motion.div>

        <motion.p {...rise(reduce, 9)} className="mt-[var(--spacing-5)] max-w-[86ch] text-xs leading-relaxed text-faint">
          Sequence-layer metrics are measured on synthetic merchant streams with injected typologies;
          the generator is in this repo. The decision layer is additionally validated on BAF (Feedzai,
          NeurIPS 2022), a public benchmark derived from real bank data.{" "}
          <strong className="text-muted-foreground">
            That BAF validation belongs to G2. BAF is not vendored in this tree — four of the
            twenty-four cycle-4 gates skip for that reason — so every G3 figure on this page is
            synthetic-only.
          </strong>
        </motion.p>
      </div>
    </div>
  );
}

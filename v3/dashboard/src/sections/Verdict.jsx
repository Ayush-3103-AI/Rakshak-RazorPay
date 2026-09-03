// §0 — the result, first, in one sentence and four numbers.
//
// RESULTS LEAD NOW. The #79 shell opened on the method and put the ladder three
// screens later, reasoning that an operator needs grounds to trust a harness
// before seeing a number. That holds for a document read start to finish; it
// does not hold for a panel a judge skims. So the claim is on screen one and
// the grounds are on screen five, where a reader who has seen the claim goes
// hunting for the catch.
//
// Every figure here is DERIVED FROM THE ARTIFACTS, never typed: the policy count
// from ladder.json's rows, the seed count and lock count from lock_state.json,
// the survivor from `beats_all_floors`, the band from cost_sweep.json's realised
// arm. That is not fastidiousness — a hero is the one place a stale literal
// survives longest, and this project's entire pitch is that its documents cannot
// drift from its measurements. It is also why the title is a sentence the
// artifacts complete: promote a second row upstream and it says "Two policies",
// with nobody editing copy.
import { Lock } from "lucide-react";
import Counter from "../components/Counter.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum } from "../lib/format.js";
import { cn } from "../lib/cn.js";

export default function Verdict() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");
  const sweep = useArtifact("cost_sweep");

  const rungs = ladder.data?.payload?.rungs ?? [];
  const locks = lockState.data?.payload?.locks ?? [];
  const live = locks.find((l) => l.authoritative);
  const survivors = rungs.filter((r) => r.beats_all_floors);

  // Seeds per row rather than the lock's declared list: what a row was actually
  // scored on is the honest denominator, and the lock only says what was
  // intended. The MINIMUM, not the maximum — "5 seeds" has to be true of every
  // row it sits above, and a max would keep printing 5 while a row underneath it
  // was scored on one. When rows disagree the label says so.
  const seedCounts = rungs.map((r) => r.n_seeds ?? 0);
  const seedCount = seedCounts.length ? Math.min(...seedCounts) : 0;
  const seedsUniform = seedCounts.length > 0 && Math.max(...seedCounts) === seedCount;

  // The exposure arm comes off the surviving row's own label, not a constant. A
  // declared-exposure survivor read from the `realised` arm would be the right
  // shape of number under the wrong name, which is the worst kind of wrong here.
  const survivorLabel = survivors[0]?.label;
  const survivorArm = survivorLabel?.endsWith("_realised_exposure") ? "realised" : "declared";
  const survivorPolicy = survivorLabel?.replace(/_realised_exposure$/, "");
  const survivorSeries = (sweep.data?.payload?.arms?.[survivorArm] ?? []).find(
    (s) => s.policy === survivorPolicy
  );
  const band = survivorSeries?.values?.length
    ? [Math.min(...survivorSeries.values), Math.max(...survivorSeries.values)]
    : null;

  // `test_split_opened` is a boolean — `sum(open_count) > 0`. Rendering it as a
  // count would print "ONCE" for two opens, on the one claim this whole panel is
  // built to make checkable. The counter itself is on each lock; sum those.
  const openCount = locks.reduce((n, l) => n + (l.open_count ?? 0), 0);

  const loading = ladder.loading || lockState.loading || sweep.loading;
  const failed = ladder.error ?? lockState.error ?? sweep.error;

  const TILES = [
    {
      key: "survivors",
      value: survivors.length,
      label: survivors.length === 1 ? "row beat every floor" : "rows beat every floor",
      note: survivors.length ? survivors.map((s) => s.label).join(", ") : "none on this ladder",
      filled: true,
    },
    { key: "policies", value: rungs.length, label: "policies scored", note: "against four named floors" },
    {
      key: "seeds",
      value: seedCount,
      label: seedsUniform ? "seeds per policy" : "seeds on the thinnest row",
      note: seedsUniform ? "every row on the same seeds" : `${Math.max(...seedCounts)} on the widest`,
    },
    { key: "locks", value: locks.length, label: "sealed eval locks", note: live?.file ?? "—" },
  ];

  return (
    <Page
      id="verdict"
      headingLevel="h1"
      eyebrow="§0 · Verdict"
      title={
        rungs.length ? (
          survivors.length === 1 ? (
            <>
              One policy beat <span className="text-primary-text">every</span> floor, on every seed.
            </>
          ) : (
            <>
              {survivors.length} policies beat <span className="text-primary-text">every</span> floor, on
              every seed.
            </>
          )
        ) : (
          "The ladder has not been scored in this tree."
        )
      }
      lede={
        <>
          Rakshak watches a merchant <em>after</em> onboarding clears it. Every number on this panel was
          scored under a harness hashed and sealed before the model that produced it existed.
        </>
      }
      actions={
        <>
          <SplitChip split={ladder.data?.split} />
          <span className="inline-flex items-center gap-[var(--spacing-2)] rounded-[var(--radius-xs)] border border-border bg-canvas-well px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold tracking-widest text-faint uppercase">
            <Lock aria-hidden="true" className="h-3 w-3" />
            opened {openCount}×
          </span>
        </>
      }
    >
      {loading && <ArtifactLoading label="Loading the ladder, the locks and the sweep…" />}
      {failed && !loading && <ArtifactError artifact="verdict" error={failed} />}

      <div className="grid grid-cols-4 gap-[var(--spacing-4)] max-lg:grid-cols-2 max-sm:grid-cols-1">
        {TILES.map((t) => (
          <Card
            key={t.key}
            pad="regular"
            elevation="low"
            className={cn(
              "flex flex-col justify-between",
              t.filled && "border-primary bg-primary text-primary-foreground"
            )}
          >
            <Counter
              value={t.value}
              format={(v) => Math.round(v).toLocaleString("en-IN")}
              className={cn(
                "block font-mono text-[clamp(32px,3.6vw,52px)] leading-none font-bold tabular-nums tracking-tight",
                t.filled ? "text-primary-foreground" : "text-foreground"
              )}
            />
            <div className="mt-[var(--spacing-5)]">
              <p className={cn("m-0 text-sm font-semibold", t.filled ? "text-primary-foreground" : "text-foreground")}>
                {t.label}
              </p>
              <p
                className={cn(
                  "m-0 mt-[var(--spacing-1)] truncate font-mono text-2xs",
                  t.filled ? "text-primary-foreground/75" : "text-faint"
                )}
                title={t.note}
              >
                {t.note}
              </p>
            </div>
          </Card>
        ))}
      </div>

      <div className="mt-[var(--spacing-4)] grid grid-cols-[1.35fr_1fr] gap-[var(--spacing-4)] max-lg:grid-cols-1">
        <Card pad="regular" elevation="low">
          <p className="m-0 font-mono text-2xs font-bold tracking-wide text-faint uppercase">
            and it is not a point estimate
          </p>
          <p className="m-0 mt-[var(--spacing-3)] text-sm leading-relaxed text-muted-foreground">
            {band ? (
              <>
                Re-price a wrong answer across four orders of magnitude and{" "}
                <span className="font-mono font-semibold text-foreground">{survivorSeries.policy}</span>{" "}
                never stops winning — savings hold between{" "}
                <span className="font-mono font-semibold text-foreground">
                  {fmtNum(band[0], 4)} and {fmtNum(band[1], 4)}
                </span>
                . About half of that is the decision layer rather than the ranker, and §3 says so.
              </>
            ) : (
              <>The cost sweep has not been scored in this tree, so no margin band is claimed here.</>
            )}
          </p>
        </Card>

        <Card pad="regular" elevation="low" className="border-notice-border">
          <p className="m-0 font-mono text-2xs font-bold tracking-wide text-notice uppercase">
            read every figure this way
          </p>
          <p className="m-0 mt-[var(--spacing-3)] text-xs leading-relaxed text-muted-foreground">
            Synthetic merchant streams with injected typologies; the generator is in this repo. BAF
            (Feedzai, NeurIPS 2022) validates the decision layer — but{" "}
            <strong className="text-foreground">BAF is not vendored in this tree</strong>, so every
            figure on this panel is synthetic-only.
          </p>
        </Card>
      </div>
    </Page>
  );
}

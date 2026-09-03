// Screen 3 — what Rakshak actually does, every day.
//
// Three words, a hard cap, and a reason on every non-PASS. The figures on the
// stat strip are the operating scale the artifacts record — the capacity K,
// the merchants and merchant-days the headline run scored, the generated
// dataset — and none of them is typed here.
import { ArtifactError } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { asText } from "../../lib/format.js";
import { deriveLadder } from "../derive.js";
import Pipeline from "../figures/Pipeline.jsx";
import Screen, { Eyebrow, Glass, Headline, Lede, Reveal, Stat } from "../Screen.jsx";

const WORDS = [
  { word: "PASS", tone: "text-faint", gloss: "Nothing to do today." },
  { word: "REVIEW", tone: "text-notice", gloss: "An analyst looks, with the reason in hand." },
  { word: "HOLD", tone: "text-negative", gloss: "Settlement pauses, and the merchant is told why." },
];

export default function Mechanism() {
  const sweep = useArtifact("cost_sweep");
  const ladder = useArtifact("ladder");
  const journey = useArtifact("journey");

  const meta = sweep.data?.payload?.meta ?? {};
  const derived = deriveLadder(ladder.data);
  const survivor = derived.survivors[0] ?? derived.rungs[0];
  const features = survivor?.metrics?.n_features ?? null;
  const k = Number.isFinite(meta.k) ? meta.k : derived.capacityK;

  // The generated-dataset figure is a journey literal on the generation that
  // built this tree; search every non-external generation's figures for it.
  const dataset =
    (journey.data?.payload?.generations ?? [])
      .filter((g) => !g.external)
      .flatMap((g) => g.figures ?? [])
      .find((f) => /dataset/i.test(asText(f.label))) ?? null;

  const failed = sweep.error ?? ladder.error;

  return (
    <Screen id="mechanism" pin={260}>
      {(progress) => (
        <div className="grid grid-cols-[5fr_7fr] items-center gap-[clamp(24px,4vw,64px)] max-lg:grid-cols-1">
          <div>
            <Eyebrow>The mechanism</Eyebrow>
            <Headline>
              Every day, for every cleared merchant, <span className="text-primary-text">one of three words.</span>
            </Headline>
            <Lede>
              Chosen under a hard analyst-capacity budget, so the system can never ask for more attention than the
              team has. Every non-PASS carries a reason a merchant can read.
            </Lede>
            <Reveal as="ul" className="m-0 mt-[var(--spacing-8)] grid list-none gap-[var(--spacing-3)] p-0">
              {WORDS.map((w) => (
                <li key={w.word} className="flex items-baseline gap-[var(--spacing-4)]">
                  <span className={`w-[7ch] shrink-0 font-mono text-sm font-bold tracking-[0.16em] ${w.tone}`}>{w.word}</span>
                  <span className="text-[length:var(--text-body)] leading-snug text-muted-foreground">{w.gloss}</span>
                </li>
              ))}
            </Reveal>
          </div>

          <div className="grid gap-[var(--spacing-4)]">
            {failed && <ArtifactError artifact="mechanism" error={failed} />}
            <Glass className="p-[clamp(12px,1.6vw,24px)]">
              <Pipeline progress={progress} k={k} features={features} />
            </Glass>
            <Glass flat className="grid gap-[var(--spacing-6)]">
              <div className="grid grid-cols-3 gap-[var(--spacing-5)] max-sm:grid-cols-1">
                <Stat value={Number.isFinite(k) ? k : "—"} label="alerts a day" note="the hard cap, K" size="xs" accent />
                <Stat value={Number.isFinite(meta.n_merchants) ? meta.n_merchants : "—"} label="merchants scored" note="headline validation run" size="xs" />
                <Stat value={Number.isFinite(meta.n_rows) ? meta.n_rows : "—"} label="merchant-days" note="one decision each" size="xs" />
              </div>
              {/* The generated dataset is a journey literal ("40,000 merchants ×
                  365 days") — a sentence, not a figure, so it reads as a line
                  rather than fighting the tiles above for a numeral slot. */}
              {dataset && (
                <p className="m-0 border-t border-border pt-[var(--spacing-5)] text-sm leading-snug text-faint">
                  <span className="font-mono font-bold text-foreground">{asText(dataset.value)}</span> generated
                  {dataset.note ? ` · ${asText(dataset.note)}` : ""}
                </p>
              )}
            </Glass>
          </div>
        </div>
      )}
    </Screen>
  );
}

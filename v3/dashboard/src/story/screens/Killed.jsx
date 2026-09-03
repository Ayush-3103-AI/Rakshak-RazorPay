// Screen 6 — what lost, on the record.
//
// Framed as discipline, because it is: a rung that loses is a finding, not an
// embarrassment, and a project that only shows wins has not measured
// anything. Every card's title and verdict are the roster's own words; the
// one-line claim above each is editorial and carries no number.
//
// The confounder null joins the list when it fires. It is not a rung, it is a
// falsified charter claim, and that belongs here more than any rung does.
import { useState } from "react";
import { ArtifactError, ArtifactLoading } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { cn } from "../../lib/cn.js";
import { deriveKilled } from "../derive.js";
import Screen, { Chip, Eyebrow, Glass, Headline, Lede, Reveal } from "../Screen.jsx";

// What each approach would have claimed, in a line. Editorial framing only:
// no measurement lives here, and an entry with no line still renders.
const CLAIMS = {
  cohort: "Residualising against a merchant's cohort should lift ranking.",
  conformal_censoring_correction: "A censoring-aware coverage bound under HOLD.",
  onset_localisation: "Localise the day a merchant's behaviour changed.",
  tpp_hawkes_nb: "A calibrated null for the alert score from arrival times.",
  rank_cusum: "Changepoint detection on the merchant's daily rank.",
  mil_gated_attention: "Learned attention over payers should beat fixed pooling.",
  tpp_neural_intensity: "A neural intensity should calibrate better than the parametric one.",
  confounder_null: "Alerts should not fire on a platform-wide event.",
};

const TONE = { cut: "negative", "not adopted": "notice", "claim falsified": "negative" };

function Card({ entry }) {
  const [open, setOpen] = useState(false);
  return (
    <Glass flat as="li" className="flex min-w-0 flex-col gap-[var(--spacing-4)] p-[var(--spacing-7)]">
      <div className="flex flex-wrap items-center gap-[var(--spacing-3)]">
        <Chip tone={TONE[entry.kind] ?? "muted"}>{entry.kind}</Chip>
        {entry.rung != null && <span className="font-mono text-[11px] tracking-[0.14em] text-faint uppercase">rung {entry.rung}</span>}
      </div>
      <h3 className="m-0 font-heading text-lg leading-snug font-bold text-foreground">{entry.title}</h3>
      {CLAIMS[entry.name] && <p className="m-0 text-sm leading-snug text-muted-foreground">{CLAIMS[entry.name]}</p>}
      {entry.verdict && (
        <div className="mt-auto">
          <p className={cn("m-0 border-l-2 border-border-strong pl-[var(--spacing-4)] text-sm leading-relaxed text-foreground/90", !open && "line-clamp-4")}>
            {entry.verdict}
          </p>
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            className="mt-[var(--spacing-3)] cursor-pointer border-0 bg-transparent p-0 font-mono text-[11px] font-bold tracking-[0.14em] text-primary-text uppercase hover:underline"
          >
            {open ? "Less" : "Read the verdict"}
          </button>
        </div>
      )}
    </Glass>
  );
}

export default function Killed() {
  const roster = useArtifact("rung_roster");
  const g5 = useArtifact("g5_confounder_null");
  const entries = deriveKilled(roster.data, g5.data);

  return (
    <Screen id="killed">
      <Eyebrow>The discipline</Eyebrow>
      <Headline>
        What lost is published, <span className="text-primary-text">with the number that killed it.</span>
      </Headline>
      <Lede>
        A rung that loses is a finding, not an embarrassment. Each approach below was built, measured on the sealed
        harness, and dropped on the record with its own verdict. {entries.length ? `${entries.length} so far.` : ""}
      </Lede>

      {(roster.loading || g5.loading) && <ArtifactLoading label="Loading the roster and the null run…" />}
      {roster.error && <ArtifactError artifact="rung_roster" error={roster.error} />}
      {g5.error && <ArtifactError artifact="g5_confounder_null" error={g5.error} />}

      {entries.length > 0 && (
        <Reveal as="ul" className="m-0 mt-[var(--spacing-10)] grid list-none grid-cols-3 gap-[var(--spacing-4)] p-0 max-lg:grid-cols-2 max-sm:grid-cols-1">
          {entries.map((e) => (
            <Card key={e.key} entry={e} />
          ))}
        </Reveal>
      )}

      <Reveal className="mt-[var(--spacing-8)]">
        <a href="#/evidence/killed" className="font-mono text-xs font-bold tracking-[0.14em] text-primary-text uppercase no-underline hover:underline">
          Every verdict in full, on the evidence panel →
        </a>
      </Reveal>
    </Screen>
  );
}

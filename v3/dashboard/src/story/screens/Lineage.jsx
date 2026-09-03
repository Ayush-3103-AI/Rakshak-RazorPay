// Screen 7 — three generations, each starting where the last one failed.
//
// The figures are the committed literals from journey.json, because prior
// harnesses are closed forever and no G1 or G2 number is ever recomputed.
// G1 lives in a different tree and is marked as such, in its own register:
// the rupee figure on this screen is real, is G1's, and travels with the
// words "cited, not recomputed" wherever it goes. That is the only place on
// the front door it appears.
import { ExternalLink } from "lucide-react";
import { ArtifactError, ArtifactLoading } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { asText } from "../../lib/format.js";
import { cn } from "../../lib/cn.js";
import { deriveJourney } from "../derive.js";
import Screen, { Chip, Eyebrow, Glass, Headline, Lede, Reveal } from "../Screen.jsx";

export default function Lineage() {
  const journey = useArtifact("journey");
  const { generations, span } = deriveJourney(journey.data);

  return (
    <Screen id="lineage">
      <Eyebrow>The lineage{span ? ` · ${span}` : ""}</Eyebrow>
      <Headline>
        Three generations. <span className="text-primary-text">Each began where the last one failed.</span>
      </Headline>
      <Lede>
        Not one build polished for a demo. Each generation started from the previous one&rsquo;s falsification rather
        than its success, and the numbers that ended each are kept as they were, not rewritten.
      </Lede>

      {journey.loading && <ArtifactLoading label="Loading journey.json…" />}
      {journey.error && <ArtifactError artifact="journey" error={journey.error} />}

      {generations.length > 0 && (
        <Reveal as="ol" className="m-0 mt-[var(--spacing-10)] grid list-none grid-cols-3 gap-[var(--spacing-4)] p-0 max-lg:grid-cols-1">
          {generations.map((g) => (
            <Glass
              flat
              as="li"
              key={g.id}
              className={cn("flex min-w-0 flex-col gap-[var(--spacing-4)] border-t-4 p-[var(--spacing-7)]", g.external ? "border-t-notice" : "border-t-primary")}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-[var(--spacing-3)]">
                <span className="font-mono text-[length:var(--text-stat-sm)] leading-none font-bold tracking-[-0.03em] text-primary-text">
                  {g.id}
                </span>
                {g.era && <span className="font-mono text-[11px] tracking-[0.14em] text-faint uppercase">{asText(g.era)}</span>}
              </div>
              <h3 className="m-0 font-heading text-xl leading-snug font-bold text-foreground">{asText(g.title)}</h3>
              {g.external && (
                <Chip tone="notice" className="self-start">
                  <ExternalLink aria-hidden="true" className="h-3 w-3" />
                  {asText(g.provenance_note)}
                </Chip>
              )}
              <p className="m-0 line-clamp-5 text-sm leading-relaxed text-muted-foreground">
                <strong className="font-semibold text-foreground">Outcome.</strong> {asText(g.outcome)}
              </p>
              {g.figures?.length > 0 && (
                <dl className="m-0 mt-auto grid gap-[var(--spacing-2)] pt-[var(--spacing-3)]">
                  {g.figures.slice(0, 3).map((f) => (
                    <div key={asText(f.label)} className="flex items-baseline justify-between gap-[var(--spacing-3)] border-t border-border pt-[var(--spacing-3)]">
                      <dt className="m-0 min-w-0 truncate text-xs text-faint" title={asText(f.label)}>
                        {asText(f.label)}
                      </dt>
                      <dd className="m-0 shrink-0 font-mono text-sm font-bold tabular-nums text-foreground">{asText(f.value)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </Glass>
          ))}
        </Reveal>
      )}
    </Screen>
  );
}

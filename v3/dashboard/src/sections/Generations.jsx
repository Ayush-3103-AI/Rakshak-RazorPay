// §1 — the lineage, as three cards side by side.
//
// This used to be a stacked <ol> of full-width cards carrying two paragraphs of
// framing before the first generation appeared. Three abreast is the honest
// shape: the point of this screen is the COMPARISON — each generation began at
// the previous one's falsification — and a comparison you have to scroll to make
// is a comparison most readers will not make.
//
// The figures are still committed literals from journey.json (built from
// configs/journey.yaml, which refuses an entry that cites nothing), because
// Prime Directive 2 closes prior harnesses forever and no G1 or G2 number is
// ever recomputed. G1 lives in a different repository and keeps its own marker
// and its own register: a number this tree cannot regenerate must not sit in the
// same visual weight as one it can.
import { ExternalLink, ScrollText } from "lucide-react";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { asText } from "../lib/format.js";
import { cn } from "../lib/cn.js";

export default function Generations() {
  const journey = useArtifact("journey");
  const generations = journey.data?.payload?.generations ?? [];

  return (
    <Page
      id="lineage"
      eyebrow="§1 · Lineage"
      title="Two generations had to fail first."
      lede={
        <>
          Nothing watches a merchant that already cleared onboarding — bust-outs and laundering begin{" "}
          <em>after</em> approval and surface 45–120 days later. Three attempts at that gap, each
          starting from the last one's falsification rather than its success.
        </>
      }
    >
      {journey.loading && <ArtifactLoading label="Loading journey.json…" />}
      {journey.error && <ArtifactError artifact="journey" error={journey.error} />}

      {generations.length > 0 && (
        <ol className="m-0 grid list-none grid-cols-3 gap-[var(--spacing-4)] p-0 max-lg:grid-cols-1">
          {generations.map((g) => (
            <li key={g.id} className="min-w-0">
              <Card
                pad="regular"
                elevation="low"
                className={cn(
                  "flex h-full flex-col border-t-4",
                  g.external ? "border-t-notice-border" : "border-t-primary"
                )}
              >
                <div className="flex flex-wrap items-baseline gap-[var(--spacing-3)]">
                  <span className="font-mono text-2xl font-extrabold tracking-tight text-primary-text">
                    {g.id}
                  </span>
                  {g.era && <span className="font-mono text-2xs text-faint">{asText(g.era)}</span>}
                </div>

                <h3 className="m-0 mt-[var(--spacing-2)] font-heading text-base leading-snug font-bold text-foreground">
                  {asText(g.title)}
                </h3>

                {g.external && (
                  <p className="m-0 mt-[var(--spacing-3)] inline-flex items-center gap-[var(--spacing-2)] rounded-[var(--radius-xs)] border border-notice-border bg-notice-bg px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold tracking-wide text-notice uppercase">
                    <ExternalLink aria-hidden="true" className="h-3 w-3 shrink-0" />
                    {asText(g.provenance_note)}
                  </p>
                )}

                <p className="m-0 mt-[var(--spacing-3)] line-clamp-2 text-xs leading-relaxed text-faint">
                  {asText(g.thesis)}
                </p>

                <p className="m-0 mt-[var(--spacing-4)] text-sm leading-relaxed text-foreground">
                  <strong className="font-semibold">Outcome.</strong> {asText(g.outcome)}
                </p>

                {g.figures?.length > 0 && (
                  <dl className="m-0 mt-[var(--spacing-5)] grid gap-[var(--spacing-2)]">
                    {g.figures.map((f) => (
                      <div
                        key={asText(f.label)}
                        className="flex items-baseline justify-between gap-[var(--spacing-3)] rounded-[var(--radius-sm)] border border-border bg-canvas-well px-[var(--spacing-4)] py-[var(--spacing-3)]"
                      >
                        <dt className="m-0 min-w-0 truncate text-2xs text-faint" title={asText(f.label)}>
                          {asText(f.label)}
                        </dt>
                        <dd className="m-0 shrink-0 font-mono text-sm font-bold tabular-nums text-foreground">
                          {asText(f.value)}
                        </dd>
                      </div>
                    ))}
                  </dl>
                )}

                <p className="m-0 mt-auto flex items-start gap-[var(--spacing-2)] pt-[var(--spacing-5)] font-mono text-2xs text-faint">
                  <ScrollText aria-hidden="true" className="mt-[2px] h-3 w-3 shrink-0" />
                  <span className="min-w-0 truncate" title={asText(g.citation)}>
                    {asText(g.citation)}
                  </span>
                </p>
              </Card>
            </li>
          ))}
        </ol>
      )}
    </Page>
  );
}

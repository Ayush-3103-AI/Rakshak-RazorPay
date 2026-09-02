// §1 — the gap, and the three generations that have tried to fill it (#79).
//
// This section replaces the hand-written WAYPOINTS array that used to live in
// TrajectoryLadder.jsx. Those were literals for a good reason — Prime Directive 2
// closes prior harnesses forever, so no G1 or G2 number is ever recomputed — but
// they were literals in a VIEW, where nothing could check them. They now come
// from journey.json, which is built from configs/journey.yaml and refuses an
// entry or a figure that cites nothing. Same immutability, checkable provenance.
//
// G1 lives in a different repository. It is rendered with its own marker and its
// own repo link, because a number this tree cannot regenerate must not sit in the
// same visual register as one it can.
import { motion, useReducedMotion } from "framer-motion";
import { ArrowUpRight, ExternalLink, ScrollText } from "lucide-react";
import Card from "../components/ui/Card.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { asText } from "../lib/format.js";
import { cn } from "../lib/cn.js";

export default function Generations() {
  const journey = useArtifact("journey");
  const reduce = useReducedMotion();

  const payload = journey.data?.payload;
  const generations = payload?.generations ?? [];

  return (
    <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          §1 · The gap &amp; three generations
        </p>
        <h2 className="m-0 max-w-[30ch] font-heading text-3xl font-bold tracking-tight text-foreground">
          Nothing watches a merchant that already cleared onboarding
        </h2>

        <div className="mt-[var(--spacing-5)] grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-[var(--spacing-5)]">
          <p className="m-0 max-w-[68ch] text-base leading-relaxed text-muted-foreground">
            Razorpay's <strong className="text-foreground">Vulcan</strong> scores every{" "}
            <em>transaction</em> in milliseconds. Razorpay's <strong className="text-foreground">Bumblebee</strong>{" "}
            reviews every <em>merchant</em> once, at onboarding. Between them sits a gap the width of a
            merchant's entire life on the platform: bust-outs, laundering endpoints, category drift and
            refund collusion all begin <em>after</em> approval, and surface only when chargebacks land
            45–120 days later.
          </p>
          <p className="m-0 max-w-[68ch] text-base leading-relaxed text-muted-foreground">
            Three generations have attacked that gap, each starting from the previous one's
            falsification rather than its success. The labels below are public names: the internal
            charter calls the last two <code className="font-mono text-sm">v1</code> and{" "}
            <code className="font-mono text-sm">v2</code>, which collides with the sibling repository
            where <code className="font-mono text-sm">v1</code> means something else entirely. The
            mapping travels on the artifact so the two vocabularies reconcile.
          </p>
        </div>

        {journey.loading && <ArtifactLoading label="Loading journey.json…" />}
        {journey.error && <ArtifactError artifact="journey" error={journey.error} />}

        {generations.length > 0 && (
          <ol className="mt-[var(--spacing-8)] m-0 flex list-none flex-col gap-[var(--spacing-5)] p-0">
            {generations.map((g, i) => (
              <motion.li
                key={g.id}
                initial={reduce ? false : { opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ delay: reduce ? 0 : i * 0.09, duration: 0.5, ease: [0, 0, 0.2, 1] }}
              >
                <Card
                  pad="spacious"
                  elevation="low"
                  className={cn(
                    "border-l-4",
                    g.external ? "border-l-notice-border" : "border-l-primary/50"
                  )}
                >
                  <div className="flex flex-wrap items-baseline gap-[var(--spacing-3)]">
                    <span className="font-mono text-2xl font-extrabold tracking-tight text-primary-text">
                      {g.id}
                    </span>
                    <h3 className="m-0 font-heading text-xl font-bold text-foreground">{asText(g.title)}</h3>
                    {g.charter_name && (
                      <span className="rounded-full border border-border bg-canvas-well px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs text-faint">
                        charter calls this “{asText(g.charter_name)}”
                      </span>
                    )}
                    {g.era && <span className="font-mono text-2xs text-faint">{asText(g.era)}</span>}
                  </div>

                  {g.external && (
                    <p className="m-0 mt-[var(--spacing-3)] inline-flex flex-wrap items-center gap-[var(--spacing-2)] rounded-[var(--radius-xs)] border border-notice-border bg-notice-bg px-[var(--spacing-3)] py-[var(--spacing-2)] font-mono text-2xs font-bold tracking-wide text-notice uppercase">
                      <ExternalLink aria-hidden="true" className="h-3 w-3 shrink-0" />
                      {asText(g.provenance_note)} — different repository
                      {g.source_repo && (
                        <a
                          href={g.source_repo}
                          className="inline-flex items-center gap-[2px] underline underline-offset-2"
                          rel="noreferrer noopener"
                          target="_blank"
                        >
                          {String(g.source_repo).replace("https://github.com/", "")}
                          <ArrowUpRight aria-hidden="true" className="h-3 w-3" />
                        </a>
                      )}
                    </p>
                  )}

                  <p className="m-0 mt-[var(--spacing-4)] max-w-[80ch] text-sm leading-relaxed text-muted-foreground">
                    {asText(g.thesis)}
                  </p>
                  <p className="m-0 mt-[var(--spacing-3)] max-w-[80ch] text-sm leading-relaxed text-foreground">
                    <strong>Outcome.</strong> {asText(g.outcome)}
                  </p>

                  {g.figures?.length > 0 && (
                    <dl className="mt-[var(--spacing-5)] m-0 grid grid-cols-[repeat(auto-fit,minmax(210px,1fr))] gap-[var(--spacing-4)]">
                      {g.figures.map((f) => (
                        <div
                          key={asText(f.label)}
                          className="rounded-[var(--radius-sm)] border border-border bg-canvas-well p-[var(--spacing-4)]"
                        >
                          <dt className="m-0 text-2xs font-semibold tracking-wide text-faint uppercase">
                            {asText(f.label)}
                          </dt>
                          <dd className="m-0 mt-[var(--spacing-1)] font-mono text-lg font-bold tabular-nums text-foreground">
                            {asText(f.value)}
                          </dd>
                          {f.note && (
                            <p className="m-0 mt-[var(--spacing-2)] text-2xs leading-relaxed text-muted-foreground">
                              {asText(f.note)}
                            </p>
                          )}
                          <p className="m-0 mt-[var(--spacing-2)] flex items-start gap-[var(--spacing-1)] font-mono text-2xs text-faint">
                            <ScrollText aria-hidden="true" className="mt-[2px] h-3 w-3 shrink-0" />
                            {asText(f.citation)}
                          </p>
                        </div>
                      ))}
                    </dl>
                  )}

                  <p className="m-0 mt-[var(--spacing-5)] flex items-start gap-[var(--spacing-2)] font-mono text-2xs text-faint">
                    <ScrollText aria-hidden="true" className="mt-[2px] h-3 w-3 shrink-0" />
                    {asText(g.citation)}
                  </p>
                </Card>
              </motion.li>
            ))}
          </ol>
        )}

        {payload?.naming?.note && (
          <p className="mt-[var(--spacing-6)] max-w-[86ch] text-xs leading-relaxed text-faint">
            {asText(payload.naming.note)}
          </p>
        )}
      </div>
    </div>
  );
}

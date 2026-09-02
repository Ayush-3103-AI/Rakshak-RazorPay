// #64 — Rungs 5-8. The ladder is built from scored EvalResult rows, so a rung
// that was never scored is INVISIBLE to it: not shown as deferred, simply
// absent. This section is the opposite of that, and it is deliberately not an
// apology — a capability that was specified, gated behind a lock, and left
// unscored on purpose is a statement about discipline, not a gap.
//
// Per #64's AC this is the one place where a missing artefact removes the
// section rather than rendering an error: if rung_roster.json is absent, the
// section renders nothing at all.
import { motion, useReducedMotion } from "framer-motion";
import { Ban, FileText, Lock, ShieldQuestion } from "lucide-react";
import Card from "../components/ui/Card.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { asText } from "../lib/format.js";
import { cn } from "../lib/cn.js";

// Editorial framing per capability — what the rung would claim and where it
// sits in the cascade. No measurement lives here; every status, reason,
// citation and note on screen is read from the roster artefact.
const CAPABILITIES = [
  {
    rung: 5,
    name: "MIL",
    title: "Noisy-OR / log-sum-exp pooling over payer capsules",
    claim: "Would place a fitted τ on the axis between max-pooling and mean-pooling, showing which extreme merchant-level evidence actually behaves like.",
    artefact: "the fitted τ against the pooling extremes it was tested to recover",
    role: null,
  },
  {
    rung: 6,
    name: "Conformal",
    title: "Mondrian conformal risk control over the three-action decision",
    claim: "Would put a distribution-free bound under the HOLD action: nominal α against realised coverage, per stratum.",
    artefact: "the coverage plot, with the censoring-weighted correction beside the uncorrected version",
    role: null,
  },
  {
    rung: 7,
    name: "HSMM",
    title: "HSMM with negative-binomial emissions",
    claim: "Would segment an alerted merchant's history into named phases and localise the day behaviour changed.",
    artefact: "a segmented replay of an alerted merchant, with the onset-localisation error distribution",
    role: "Stage-2 explainer only — runs on non-PASS decisions, never in the scoring path, never scored on PR-AUC.",
  },
  {
    rung: 8,
    name: "TPP",
    title: "Hawkes/NB temporal point process with time-rescaling KS",
    claim: "Would test whether inter-arrival structure carries signal the tabular features miss.",
    artefact: "the KS null distribution at prevalence 0 with confounders on, beside the in-sample rescaled-time result",
    role: null,
  },
];

function Entry({ entry }) {
  const contradicts = entry.contradicts ?? [];
  return (
    <div className="rounded-[var(--radius-sm)] border border-border bg-canvas-well p-[var(--spacing-4)]">
      <div className="flex flex-wrap items-center gap-[var(--spacing-2)]">
        <span className="font-mono text-xs font-semibold text-foreground">{entry.name}</span>
        <StatusChip status={entry.status} />
        {entry.adopted === false && (
          <span className="rounded-full border border-border px-[var(--spacing-2)] py-[1px] font-mono text-2xs text-faint uppercase">
            not adopted
          </span>
        )}
        {entry.ticket && <span className="font-mono text-2xs text-faint">{entry.ticket}</span>}
        {entry.issue && <span className="font-mono text-2xs text-faint">#{entry.issue}</span>}
      </div>

      {entry.title && <p className="m-0 mt-[var(--spacing-2)] text-sm text-foreground">{asText(entry.title)}</p>}
      {entry.reason && (
        <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-muted-foreground">
          <strong className="text-foreground">Why: </strong>
          {asText(entry.reason)}
        </p>
      )}
      {entry.note && (
        <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-muted-foreground">{asText(entry.note)}</p>
      )}
      {entry.blocked_by && (
        <p className="m-0 mt-[var(--spacing-2)] flex items-start gap-[var(--spacing-2)] text-xs leading-relaxed text-notice">
          <Lock aria-hidden="true" className="mt-[2px] h-3 w-3 shrink-0" />
          {asText(entry.blocked_by)}
        </p>
      )}
      {entry.decided_in && (
        <p className="m-0 mt-[var(--spacing-2)] text-xs text-muted-foreground">
          <strong className="text-foreground">Decided in: </strong>
          {asText(entry.decided_in)}
        </p>
      )}
      {contradicts.length > 0 && (
        <div className="mt-[var(--spacing-3)] rounded-[var(--radius-xs)] border border-notice-border bg-notice-bg p-[var(--spacing-3)]">
          <p className="m-0 font-mono text-2xs font-bold tracking-wide text-notice uppercase">
            standing contradiction in the committed docs
          </p>
          <ul className="m-0 mt-[var(--spacing-2)] list-none space-y-[2px] p-0">
            {contradicts.map((c) => (
              <li key={asText(c)} className="text-2xs leading-relaxed text-muted-foreground">
                {asText(c)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {entry.gap && (
        <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-notice">{asText(entry.gap)}</p>
      )}
      <ul className="m-0 mt-[var(--spacing-3)] list-none space-y-[2px] p-0">
        {(entry.citation ?? []).map((c) => (
          <li key={asText(c)} className="flex items-start gap-[var(--spacing-2)] font-mono text-2xs text-faint">
            <FileText aria-hidden="true" className="mt-[2px] h-3 w-3 shrink-0" />
            {asText(c)}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function DeferredRungs() {
  const roster = useArtifact("rung_roster");
  const reduce = useReducedMotion();

  // A missing roster removes the section — the one place "missing" is
  // expected rather than a fault (#64). Loading still shows, so an in-flight
  // fetch is never mistaken for an absent capability.
  if (roster.error) return null;

  const payload = roster.data?.payload;
  const entries = payload?.roster ?? [];

  return (
    <div className="px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          §4 · What we killed
        </p>
        <h2 className="m-0 max-w-[30ch] font-heading text-3xl font-bold tracking-tight text-foreground">
          Specified, gated, and either unscored or not adopted
        </h2>
        <p className="mt-[var(--spacing-4)] max-w-[70ch] text-base leading-relaxed text-muted-foreground">
          This section is the counterweight to §3, and it is deliberately not an apology. Prime Directive 6
          says a rung that loses is a finding: four new rungs were built and scored on real cycle-4 data and
          all four were <strong className="text-foreground">NOT ADOPTED</strong>, with the numbers written
          down. Other capabilities sit behind a lock unscored on purpose — a rung landing after the lock is
          post-lock and ineligible for adoption, so shipping one early would cost more than it bought. Each
          entry below says what it is, what it would have claimed, and exactly what happened to it, read
          from the roster, which carries a citation for every line.
        </p>

        {roster.loading && (
          <div className="mt-[var(--spacing-6)]">
            <ArtifactLoading label="Loading rung_roster.json…" />
          </div>
        )}

        {payload && (
          <>
            <div className="mt-[var(--spacing-7)] grid gap-[var(--spacing-5)]">
              {CAPABILITIES.map((cap, i) => {
                const rows = entries.filter((e) => e.rung === cap.rung);
                if (!rows.length) return null;
                const allCut = rows.every((r) => r.status === "cut");
                return (
                  <motion.div
                    key={cap.name}
                    initial={reduce ? false : { opacity: 0, y: 16 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-50px" }}
                    transition={{ delay: reduce ? 0 : i * 0.06, duration: 0.45, ease: [0, 0, 0.2, 1] }}
                  >
                    <Card pad="regular" elevation="low" className={cn(allCut && "border-negative-border")}>
                      <div className="flex flex-wrap items-center gap-[var(--spacing-3)]">
                        <span className="tabular rounded-[var(--radius-xs)] border border-border px-[var(--spacing-3)] py-[1px] font-mono text-2xs font-bold text-faint">
                          RUNG {cap.rung}
                        </span>
                        <h3 className="m-0 font-heading text-lg font-semibold text-foreground">{cap.name}</h3>
                        <span className="text-sm text-muted-foreground">{cap.title}</span>
                        {allCut && (
                          <span className="inline-flex items-center gap-[var(--spacing-1)] rounded-full border border-negative-border bg-negative-bg px-[var(--spacing-3)] py-[1px] font-mono text-2xs font-bold text-negative uppercase">
                            <Ban aria-hidden="true" className="h-3 w-3" />
                            cut
                          </span>
                        )}
                      </div>

                      <p className="m-0 mt-[var(--spacing-3)] max-w-[76ch] text-sm leading-relaxed text-muted-foreground">
                        {cap.claim}
                      </p>

                      {cap.role && (
                        <p className="m-0 mt-[var(--spacing-3)] inline-flex items-start gap-[var(--spacing-2)] rounded-[var(--radius-xs)] border border-information-border bg-information-bg px-[var(--spacing-3)] py-[var(--spacing-2)] text-xs font-medium text-information">
                          <ShieldQuestion aria-hidden="true" className="mt-[1px] h-3.5 w-3.5 shrink-0" />
                          {cap.role}
                        </p>
                      )}

                      <div className="mt-[var(--spacing-4)] grid gap-[var(--spacing-3)]">
                        {rows.map((entry) => (
                          <Entry key={`${entry.name}-${entry.ticket ?? ""}`} entry={entry} />
                        ))}
                      </div>

                      <p className="m-0 mt-[var(--spacing-4)] border-t border-border pt-[var(--spacing-3)] text-xs text-faint">
                        When this rung lands it publishes {cap.artefact}, with its own split label. Until the
                        artefact exists this panel shows the roster and nothing more — there is no placeholder
                        chart here, because a chart with no measurement behind it is the one thing this
                        contract exists to prevent.
                      </p>
                    </Card>
                  </motion.div>
                );
              })}
            </div>

            <Card pad="compact" elevation="low" className="mt-[var(--spacing-6)] border-notice-border">
              <p className="m-0 flex items-center gap-[var(--spacing-2)] font-mono text-2xs font-bold tracking-wide text-notice uppercase">
                <ShieldQuestion aria-hidden="true" className="h-3.5 w-3.5" />
                roster caveat · {payload.n_unverified} UNVERIFIED{" "}
                {payload.unverified?.length ? `(${payload.unverified.join(", ")})` : ""}
              </p>
              <p className="m-0 mt-[var(--spacing-3)] max-w-[86ch] text-xs leading-relaxed text-muted-foreground">
                {asText(payload.roster_note)}
              </p>
              {payload.source?.standing_contradiction && (
                <p className="m-0 mt-[var(--spacing-3)] max-w-[86ch] text-xs leading-relaxed text-muted-foreground">
                  <strong className="text-foreground">Standing contradiction: </strong>
                  {asText(payload.source.standing_contradiction)}
                </p>
              )}
              {payload.source?.known_gap && (
                <p className="m-0 mt-[var(--spacing-3)] max-w-[86ch] text-xs leading-relaxed text-muted-foreground">
                  <strong className="text-foreground">Known gap: </strong>
                  {asText(payload.source.known_gap)}
                </p>
              )}
            </Card>
          </>
        )}
      </div>
    </div>
  );
}

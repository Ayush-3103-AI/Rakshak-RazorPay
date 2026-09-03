// §6 — Rungs 5-8, and what happened to each.
//
// The ladder is built from scored EvalResult rows, so a rung that was never
// scored is INVISIBLE to it: not shown as deferred, simply absent. This screen
// is the opposite of that, and it is deliberately not an apology — a capability
// that was specified, gated behind a lock and left unscored on purpose is a
// statement about discipline, not a gap.
//
// Per #64's AC this is the one place where a missing artefact removes the
// section rather than rendering an error: if rung_roster.json is absent, the
// page renders nothing at all.
import { Ban, ShieldQuestion } from "lucide-react";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { asText } from "../lib/format.js";
import { cn } from "../lib/cn.js";

// Editorial framing per capability — what the rung would have claimed. No
// measurement lives here; every status, reason and citation on screen is read
// from the roster artefact.
const CAPABILITIES = [
  { rung: 5, name: "MIL", claim: "Where merchant-level evidence really sits between max- and mean-pooling." },
  { rung: 6, name: "Conformal", claim: "A distribution-free coverage bound under the HOLD action, per stratum." },
  { rung: 7, name: "HSMM", claim: "Segment an alerted merchant's history and localise the day behaviour changed." },
  { rung: 8, name: "TPP", claim: "Whether inter-arrival structure carries signal the tabular features miss." },
];

function Entry({ entry }) {
  return (
    <div className="rounded-[var(--radius-sm)] border border-border bg-canvas-well p-[var(--spacing-3)]">
      <div className="flex flex-wrap items-center gap-[var(--spacing-2)]">
        <span className="font-mono text-2xs font-semibold text-foreground">{entry.name}</span>
        <StatusChip status={entry.status} />
        {entry.adopted === false && (
          <span className="rounded-full border border-border px-[var(--spacing-2)] py-[1px] font-mono text-2xs text-faint uppercase">
            not adopted
          </span>
        )}
      </div>
      {entry.reason && (
        <p className="m-0 mt-[var(--spacing-2)] text-2xs leading-relaxed text-muted-foreground">
          <strong className="text-foreground">Why: </strong>
          {asText(entry.reason)}
        </p>
      )}
      {entry.blocked_by && (
        <p className="m-0 mt-[var(--spacing-2)] text-2xs leading-relaxed text-notice">
          {asText(entry.blocked_by)}
        </p>
      )}
    </div>
  );
}

export default function DeferredRungs() {
  const roster = useArtifact("rung_roster");

  // A missing roster removes the section — the one place "missing" is expected
  // rather than a fault (#64). Loading still shows, so an in-flight fetch is
  // never mistaken for an absent capability.
  if (roster.error) return null;

  const payload = roster.data?.payload;
  const entries = payload?.roster ?? [];

  return (
    <Page
      id="killed"
      eyebrow="§6 · Discipline"
      title="What we killed, with the numbers."
      lede={
        <>
          A rung that loses is a finding, not an embarrassment. Four were built and scored on real
          cycle-4 data and all four came back <strong className="text-foreground">NOT ADOPTED</strong>.
          The rest sit behind a lock, unscored on purpose.
        </>
      }
    >
      {roster.loading && <ArtifactLoading label="Loading rung_roster.json…" />}

      {payload && (
        <div className="flex h-full min-h-0 flex-col gap-[var(--spacing-4)]">
          <div className="grid min-h-0 flex-1 grid-cols-4 gap-[var(--spacing-4)] overflow-auto max-xl:grid-cols-2 max-md:grid-cols-1">
            {CAPABILITIES.map((cap) => {
              const rows = entries.filter((e) => e.rung === cap.rung);
              if (!rows.length) return null;
              const allCut = rows.every((r) => r.status === "cut");
              return (
                <Card
                  key={cap.name}
                  pad="regular"
                  elevation="low"
                  className={cn("flex h-full flex-col", allCut && "border-negative-border")}
                >
                  <div className="flex flex-wrap items-center gap-[var(--spacing-2)]">
                    <span className="rounded-[var(--radius-xs)] border border-border px-[var(--spacing-2)] py-[1px] font-mono text-2xs font-bold text-faint">
                      RUNG {cap.rung}
                    </span>
                    <h3 className="m-0 font-heading text-base font-semibold text-foreground">{cap.name}</h3>
                    {allCut && (
                      <span className="inline-flex items-center gap-[var(--spacing-1)] rounded-full border border-negative-border bg-negative-bg px-[var(--spacing-2)] py-[1px] font-mono text-2xs font-bold text-negative uppercase">
                        <Ban aria-hidden="true" className="h-3 w-3" />
                        cut
                      </span>
                    )}
                  </div>

                  <p className="m-0 mt-[var(--spacing-3)] text-xs leading-relaxed text-muted-foreground">
                    {cap.claim}
                  </p>

                  <div className="mt-[var(--spacing-4)] grid gap-[var(--spacing-2)]">
                    {rows.map((entry) => (
                      <Entry key={`${entry.name}-${entry.ticket ?? ""}`} entry={entry} />
                    ))}
                  </div>
                </Card>
              );
            })}
          </div>

          <Card pad="compact" elevation="low" className="shrink-0 border-notice-border">
            <p className="m-0 flex items-start gap-[var(--spacing-2)] text-2xs leading-relaxed text-muted-foreground">
              <ShieldQuestion aria-hidden="true" className="mt-[1px] h-3.5 w-3.5 shrink-0 text-notice" />
              <span>
                A rung landing after the lock is post-lock and ineligible for adoption, so shipping one
                early would cost more than it bought. Until a rung publishes its own artefact this panel
                shows the roster and nothing more — there is no placeholder chart here, because a chart
                with no measurement behind it is the one thing this contract exists to prevent.
              </span>
            </p>
          </Card>
        </div>
      )}
    </Page>
  );
}

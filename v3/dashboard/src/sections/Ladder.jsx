// §2 — the ladder, on a screen of its own.
//
// Split out of the old §3, which stacked the ladder, the sweep and the G5 null
// into one section. Nothing was cut in the split: this is the same table, the
// same floors and the same provenance block, given the screen it needs.
//
// The table scrolls INSIDE its card rather than growing the page. That is what
// keeps one gesture equal to one screen — a page taller than the viewport under
// mandatory snap is a page whose bottom rows you have to fight the scroller to
// reach.
import { Table2 } from "lucide-react";
import LadderTable from "../components/LadderTable.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";

export default function Ladder() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");

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

  return (
    <Page
      id="ladder"
      eyebrow="§2 · Evidence"
      title="Every policy, against the same floors."
      lede={
        <>
          A row that loses to a trivial baseline is marked as losing to <em>that named floor</em>. A
          random ranker out-earning a trained model is a result about the problem, not a fault in the
          table.
        </>
      }
      actions={
        <>
          <SplitChip split={ladder.data?.split} />
          <span className="inline-flex items-center gap-[var(--spacing-2)] font-mono text-2xs text-faint">
            <Table2 aria-hidden="true" className="h-3 w-3" />
            {ladder.data?.payload?.rungs?.length ?? 0} rows
          </span>
        </>
      }
    >
      {ladder.loading && <ArtifactLoading label="Loading ladder.json…" />}
      {ladder.error && <ArtifactError artifact="ladder" error={ladder.error} />}

      {ladder.data && (
        <div className="flex h-full min-h-0 flex-col gap-[var(--spacing-4)]">
          {/* No border and no scrolling here — LadderTable owns both, so that its
              sticky header has the scroll container it expects. This div only
              hands it a height. */}
          <div className="min-h-0 flex-1 overflow-hidden">
            <LadderTable ladder={ladder.data} windowDaysBySplit={windowDaysBySplit} />
          </div>

          {provenance && (
            <Card pad="compact" elevation="low" className="shrink-0">
              <p className="m-0 text-xs leading-relaxed text-muted-foreground">
                <strong className="text-foreground">Provenance.</strong>{" "}
                {provenance.results_are_current === false ? (
                  <>
                    At least one row was scored under a{" "}
                    <span className="font-semibold text-notice">superseded cycle</span> and is rendered as
                    such — the pre-registered state for a rung the newer cycle did not rescore, not drift.
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
            </Card>
          )}
        </div>
      )}
    </Page>
  );
}

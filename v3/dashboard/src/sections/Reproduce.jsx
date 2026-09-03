// §7 — the manifest's own PRESENT/MISSING account of every artefact this tree
// produced, plus the commands that regenerate them. Every number here reads from
// the artefact envelope — nothing is a literal.
import { FileWarning } from "lucide-react";
import LockPanel from "../components/LockPanel.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { shortSha } from "../lib/format.js";

const ARTIFACT_BLURB = {
  manifest: "Index of every artefact this tree emitted, PRESENT or MISSING, with the reason.",
  lock_state: "The eval-lock supersession chain: hashes, open counter, freezing commit.",
  ladder: "Every scored rung against the four floors and the oracle gap.",
  g5_confounder_null: "The confounder-null gate at zero prevalence — raw vs cohort-residual.",
  rung_roster: "Rungs 5-8's status where the ladder has no row to show them.",
  cost_sweep: "The savings ranking re-priced across four orders of magnitude of cost asymmetry.",
  journey: "The three generations as committed literals — G1's figures cited, not recomputed.",
};

export default function Reproduce() {
  const manifest = useArtifact("manifest");
  const lockState = useArtifact("lock_state");
  const testOpen = lockState.data?.payload?.test_split_opened;

  return (
    <Page
      id="reproduce"
      eyebrow="§7 · Provenance"
      title="Every number, and the file it came out of."
      lede={
        <>
          Read live from a committed, versioned artefact contract under{" "}
          <code className="rounded bg-canvas-well px-[5px] py-[1px] font-mono text-xs text-primary-text">
            artifacts/*.json
          </code>{" "}
          — no backend, no hardcoded figure. A missing artefact renders a named error, never a blank
          chart standing in for a number nobody measured.
        </>
      }
    >
      <div className="grid h-full min-h-0 grid-cols-[1fr_1.3fr] gap-[var(--spacing-4)] max-xl:grid-cols-1">
        <div className="flex min-h-0 flex-col gap-[var(--spacing-4)]">
          <pre className="m-0 shrink-0 overflow-x-auto rounded-[var(--radius-md)] border border-border bg-canvas-well p-[var(--spacing-5)] font-mono text-xs leading-relaxed text-muted-foreground">
{`uv sync
make all      # lint → parity → gen → gates → perf → test
make report   # regenerate docs/results_v2.md from the frozen eval

# make eval refuses the locked test split unless
# RAKSHAK_UNLOCK=1. It is not set anywhere in this repo.`}
          </pre>

          {!lockState.loading && lockState.data && !testOpen && (
            <div className="flex shrink-0 items-start gap-[var(--spacing-2)] rounded-[var(--radius-md)] border border-dashed border-notice-border bg-notice-bg px-[var(--spacing-4)] py-[var(--spacing-3)] text-xs font-medium text-notice">
              <FileWarning className="mt-[1px] h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              The test split has not opened. Every figure on this panel is VALIDATION — nothing here is
              rendered in a locked register.
            </div>
          )}

          <Card pad="regular" elevation="low" className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">Lock state</h3>
            <div className="mt-[var(--spacing-3)] min-h-0 flex-1 overflow-auto">
              {lockState.loading && <ArtifactLoading label="Loading lock_state.json…" />}
              {lockState.error && <ArtifactError artifact="lock_state" error={lockState.error} />}
              {lockState.data && <LockPanel lockState={lockState.data} variant="compact" />}
            </div>
          </Card>
        </div>

        <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
          <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">
            The artefact contract
          </h3>
          <div className="mt-[var(--spacing-3)] grid min-h-0 flex-1 grid-cols-2 content-start gap-[var(--spacing-3)] overflow-auto max-md:grid-cols-1">
            {manifest.loading && <ArtifactLoading label="Loading manifest.json…" />}
            {manifest.error && <ArtifactError artifact="manifest" error={manifest.error} />}
            {manifest.data?.payload?.artifacts?.map((a) => (
              <div
                key={a.name}
                className="rounded-[var(--radius-sm)] border border-border bg-canvas-well p-[var(--spacing-4)]"
              >
                <div className="flex items-start justify-between gap-[var(--spacing-2)]">
                  <p className="m-0 min-w-0 truncate font-mono text-2xs font-semibold text-foreground">
                    {a.name}
                  </p>
                  <StatusChip status={a.status} />
                </div>
                <p className="m-0 mt-[var(--spacing-2)] text-2xs leading-relaxed text-muted-foreground">
                  {ARTIFACT_BLURB[a.name] ?? ""}
                </p>
                <p className="m-0 mt-[var(--spacing-2)] font-mono text-2xs text-faint">
                  {a.status === "PRESENT" ? `sha256 ${shortSha(a.sha256)}` : a.reason}
                </p>
              </div>
            ))}
          </div>
          <p className="m-0 mt-[var(--spacing-3)] shrink-0 border-t border-border pt-[var(--spacing-3)] text-2xs text-faint">
            RAKSHAK G3 · data-access layer reads <code className="font-mono">artifacts/*.json</code> only.
            Built for T-0127/#61, T-0128/#62, T-0129/#63, T-0130/#64 and #79.
          </p>
        </Card>
      </div>
    </Page>
  );
}

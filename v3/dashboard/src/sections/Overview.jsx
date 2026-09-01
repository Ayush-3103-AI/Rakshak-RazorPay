// #61's shell content: the hero, the manifest's own PRESENT/MISSING account
// of every artefact this tree produced, and a compact lock summary. Every
// number here reads from the artefact envelope — nothing is a literal.
import { motion, useReducedMotion } from "framer-motion";
import { FileWarning } from "lucide-react";
import Card from "../components/ui/Card.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import LockPanel from "../components/LockPanel.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { shortSha } from "../lib/format.js";

const ARTIFACT_BLURB = {
  manifest: "Index of every artefact this tree emitted, PRESENT or MISSING, with the reason.",
  lock_state: "The eval-lock supersession chain: hashes, open counter, freezing commit.",
  ladder: "Every scored rung against the four floors and the oracle gap.",
  g5_confounder_null: "The confounder-null gate at zero prevalence — raw vs cohort-residual.",
  rung_roster: "Rungs 5-8's status where the ladder has no row to show them.",
};

function stagger(reduce) {
  return {
    hidden: { opacity: 0, y: reduce ? 0 : 14, scale: reduce ? 1 : 0.97 },
    show: (i) => ({
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { delay: reduce ? 0 : i * 0.06, duration: 0.4, ease: [0.17, 0.67, 0.3, 1.4] },
    }),
  };
}

export default function Overview() {
  const manifest = useArtifact("manifest");
  const lockState = useArtifact("lock_state");
  const reduce = useReducedMotion();
  const variants = stagger(reduce);

  const testOpen = lockState.data?.payload?.test_split_opened;

  return (
    <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          §0 · Overview
        </p>
        <h1 className="m-0 max-w-[22ch] font-heading text-4xl font-extrabold tracking-tight text-foreground md:text-5xl">
          RAKSHAK v3 — the evidence panel
        </h1>
        <p className="mt-[var(--spacing-5)] max-w-[62ch] text-lg leading-relaxed text-muted-foreground">
          Every number on this panel is read live from the committed, versioned artefact contract
          under <code className="rounded bg-canvas-well px-[6px] py-[2px] font-mono text-sm text-primary-text">artifacts/*.json</code> —
          no backend, no hardcoded figure. A missing or malformed artefact renders a named error, never a
          blank chart standing in for a number nobody measured.
        </p>

        {!lockState.loading && lockState.data && !testOpen && (
          <div className="mt-[var(--spacing-6)] inline-flex items-center gap-[var(--spacing-2)] rounded-[var(--radius-sm)] border border-dashed border-notice-border bg-notice-bg px-[var(--spacing-4)] py-[var(--spacing-3)] text-sm font-medium text-notice">
            <FileWarning className="h-4 w-4 shrink-0" aria-hidden="true" />
            The test split has not opened (open counter is 0). Every figure below is VALIDATION until
            T-0116 opens it — nothing here is rendered in a locked register.
          </div>
        )}

        <div className="mt-[var(--spacing-8)] grid grid-cols-[repeat(auto-fit,minmax(230px,1fr))] gap-[var(--spacing-4)]">
          {manifest.loading && <ArtifactLoading label="Loading manifest.json…" />}
          {manifest.error && <ArtifactError artifact="manifest" error={manifest.error} />}
          {manifest.data?.payload?.artifacts?.map((a, i) => (
            <motion.div
              key={a.name}
              custom={i}
              initial="hidden"
              whileInView="show"
              viewport={{ once: true, margin: "-40px" }}
              variants={variants}
            >
              <Card pad="compact" elevation="low">
                <div className="flex items-start justify-between gap-[var(--spacing-3)]">
                  <p className="m-0 font-mono text-xs font-semibold tracking-wide text-foreground">{a.name}</p>
                  <StatusChip status={a.status} />
                </div>
                <p className="mt-[var(--spacing-3)] mb-0 text-xs leading-relaxed text-muted-foreground">
                  {ARTIFACT_BLURB[a.name] ?? ""}
                </p>
                <p className="mt-[var(--spacing-3)] mb-0 font-mono text-2xs text-faint">
                  {a.status === "PRESENT" ? `sha256 ${shortSha(a.sha256)}` : a.reason}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>

        <div className="mt-[var(--spacing-9)]">
          <h2 className="m-0 mb-[var(--spacing-4)] font-heading text-xl font-bold text-foreground">Lock state</h2>
          {lockState.loading && <ArtifactLoading label="Loading lock_state.json…" />}
          {lockState.error && <ArtifactError artifact="lock_state" error={lockState.error} />}
          {lockState.data && <LockPanel lockState={lockState.data} variant="compact" />}
        </div>
      </div>
    </div>
  );
}

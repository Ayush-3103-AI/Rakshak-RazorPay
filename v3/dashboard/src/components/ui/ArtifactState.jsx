// The fail-visible contract, rendered: a rejected fetch, a failed response or
// a schema mismatch shows the artifact name and the reason, never an empty-
// but-plausible chart (#61's AC). Loading and MISSING are first-class states
// too, not afterthoughts bolted on once the happy path worked.
import { AlertOctagon, Loader2 } from "lucide-react";

export function ArtifactLoading({ label = "Loading artifact…" }) {
  return (
    <div className="flex items-center gap-[var(--spacing-3)] rounded-[var(--radius-md)] border border-border bg-canvas-well p-[var(--spacing-5)] text-sm text-faint">
      <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
      {label}
    </div>
  );
}

export function ArtifactError({ error, artifact }) {
  return (
    <div className="flex items-start gap-[var(--spacing-3)] rounded-[var(--radius-md)] border border-negative-border bg-negative-bg p-[var(--spacing-5)]">
      <AlertOctagon aria-hidden="true" className="mt-[2px] h-4 w-4 shrink-0 text-negative" />
      <div>
        <p className="m-0 font-mono text-2xs font-bold tracking-wide text-negative uppercase">
          {artifact ? `${artifact}: MISSING or invalid` : "Artifact unavailable"}
        </p>
        <p className="m-0 mt-[var(--spacing-2)] text-sm break-words text-foreground/80">
          {error?.message ?? String(error)}
        </p>
      </div>
    </div>
  );
}

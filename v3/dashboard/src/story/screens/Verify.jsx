// Screen 8 — don't take the page's word for it.
//
// The locks, the manifest, the commands, the links — and two lines about who
// built it. The repository link is read from journey.json rather than typed,
// so the front door cannot point somewhere the artifacts do not.
import { ExternalLink, Lock, ShieldCheck } from "lucide-react";
import { ArtifactError } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { asText } from "../../lib/format.js";
import { deriveJourney, deriveLocks, deriveManifest } from "../derive.js";
import Screen, { Eyebrow, Glass, Headline, Lede, Reveal, Stat } from "../Screen.jsx";

export default function Verify() {
  const lockState = useArtifact("lock_state");
  const manifestDoc = useArtifact("manifest");
  const journey = useArtifact("journey");
  const locks = deriveLocks(lockState.data);
  const manifest = deriveManifest(manifestDoc.data);
  const { generations, span } = deriveJourney(journey.data);
  const repo = generations.find((g) => !g.external && /^https?:/.test(asText(g.source_repo)))?.source_repo ?? null;
  const failed = lockState.error ?? manifestDoc.error;

  return (
    <Screen id="verify">
      <Eyebrow>Verify</Eyebrow>
      <Headline>
        Don&rsquo;t take this page&rsquo;s word for it.
      </Headline>
      <Lede>
        Every number above is read at load time from {manifest.total || "the"} committed artifact files, each carrying a
        sha256 and the commit and evaluation lock it was scored under. The harness was hashed and sealed before the
        models it scored existed.
      </Lede>

      {failed && <ArtifactError artifact="verify" error={failed} />}

      <div className="mt-[var(--spacing-7)] grid grid-cols-3 gap-[var(--spacing-4)] max-lg:grid-cols-1">
        <Glass className="flex flex-col gap-[var(--spacing-6)]">
          <p className="m-0 flex items-center gap-[var(--spacing-2)] font-mono text-[11px] font-bold tracking-[0.18em] text-faint uppercase">
            <Lock aria-hidden="true" className="h-3 w-3" /> the seal
          </p>
          <Stat value={locks.opens} label="times the test split was opened" note={`across ${locks.n} sealed locks`} accent />
          <div className="grid grid-cols-2 gap-[var(--spacing-5)]">
            <Stat value={locks.n} label="eval locks" note="superseding forward" size="sm" />
            <Stat value={locks.preRegistered} label="pre-registered" note="claims written before the run" size="sm" />
          </div>
        </Glass>

        <Glass className="flex flex-col gap-[var(--spacing-5)]">
          <p className="m-0 flex items-center gap-[var(--spacing-2)] font-mono text-[11px] font-bold tracking-[0.18em] text-faint uppercase">
            <ShieldCheck aria-hidden="true" className="h-3 w-3" /> the artifacts
          </p>
          <Stat value={manifest.present} label={`of ${manifest.total} artifacts present`} note="each with its sha256" size="sm" />
          <ul className="m-0 grid list-none gap-[var(--spacing-2)] p-0">
            {manifest.artifacts.map((a) => (
              <li key={a.name} className="flex items-baseline justify-between gap-[var(--spacing-3)] border-t border-border pt-[var(--spacing-2)] font-mono text-xs">
                <span className="text-foreground">{a.name}</span>
                <span className="truncate text-faint" title={a.sha256}>
                  {a.sha256?.slice(0, 12) ?? "—"}…
                </span>
              </li>
            ))}
          </ul>
        </Glass>

        <Glass className="flex flex-col gap-[var(--spacing-5)]">
          <p className="m-0 font-mono text-[11px] font-bold tracking-[0.18em] text-faint uppercase">rerun it</p>
          <pre className="m-0 overflow-x-auto rounded-[var(--radius-md)] border border-border bg-canvas-well/60 p-[var(--spacing-5)] font-mono text-xs leading-relaxed text-foreground">
            {"cd v3\nuv sync\nmake all      # from a clean clone\nmake report"}
          </pre>
          <ul className="m-0 mt-auto grid list-none gap-[var(--spacing-3)] p-0">
            <li>
              <a href="#/evidence" className="inline-flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.14em] text-primary-text uppercase no-underline hover:underline">
                The full evidence panel →
              </a>
            </li>
            {repo && (
              <>
                <li>
                  <a href={repo} target="_blank" rel="noreferrer" className="inline-flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.14em] text-primary-text uppercase no-underline hover:underline">
                    <ExternalLink aria-hidden="true" className="h-3 w-3" /> repository
                  </a>
                </li>
                <li>
                  <a href={`${repo.replace(/\/$/, "")}/blob/main/v3/LIMITATIONS.md`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-[var(--spacing-2)] font-mono text-xs font-bold tracking-[0.14em] text-primary-text uppercase no-underline hover:underline">
                    <ExternalLink aria-hidden="true" className="h-3 w-3" /> every failure, with the number
                  </a>
                </li>
              </>
            )}
          </ul>
        </Glass>
      </div>

      <Reveal className="mt-[var(--spacing-7)] grid gap-[var(--spacing-2)] border-t border-border pt-[var(--spacing-5)]">
        <p className="m-0 text-base leading-snug font-semibold text-foreground">
          Built solo by Ayush for the Razorpay AI Buildathon 2026{span ? `, ${span}` : ""}.
        </p>
        <p className="m-0 max-w-[86ch] text-xs leading-relaxed text-muted-foreground">
          Next: open the test split once, on the record, and validate the decision layer on BAF inside this tree.
          Synthetic merchant streams with injected typologies throughout; no real Razorpay data, APIs or internal
          systems were used.
        </p>
      </Reveal>
    </Screen>
  );
}

// The evidence panel's chrome: the top bar, the borderless page list, and the
// same list as chips on a narrow viewport.
//
// Both halves of the site now use the same two pieces, so navigating one
// teaches you the other. The section groups still come from the manifest
// (sections.js), and the group names ride along as the numbering — §0, §1 —
// which is what the panel numbered its screens by in the first place.
import { ArrowLeft, Lock } from "lucide-react";
import Brand from "./components/Brand.jsx";
import Rail, { RailChips } from "./components/Rail.jsx";
import TopBar from "./components/TopBar.jsx";
import { useArtifact } from "./lib/artifacts.js";
import { SECTIONS } from "./sections.js";

export default function Chrome({ activeSection, onSelect }) {
  const lockState = useArtifact("lock_state");
  const locks = lockState.data?.payload?.locks ?? [];
  const opens = locks.reduce((n, l) => n + (l.open_count ?? 0), 0);
  const activeId = SECTIONS[activeSection]?.id ?? SECTIONS[0].id;

  return (
    <>
      <TopBar
        meta={
          <span className="inline-flex items-center gap-[var(--spacing-2)]">
            <Lock aria-hidden="true" className="h-3 w-3" />
            test split opened {opens}×
          </span>
        }
        action={
          <a
            href="#/"
            className="inline-flex items-center gap-[var(--spacing-2)] rounded-full border border-border px-[var(--spacing-5)] py-[var(--spacing-3)] font-mono text-[11px] font-bold tracking-[0.14em] text-muted-foreground uppercase no-underline transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
          >
            <ArrowLeft aria-hidden="true" className="h-3 w-3" />
            Back to the story
          </a>
        }
      />

      <Rail items={SECTIONS} activeId={activeId} onSelect={onSelect} label="Panel sections" />

      <RailChips
        items={SECTIONS}
        activeId={activeId}
        onSelect={onSelect}
        label="Panel sections"
        leading={
          <>
            <Brand href="#/" size="xs" descriptor={null} className="mr-[var(--spacing-3)] shrink-0" />
            <a
              href="#/"
              className="inline-flex shrink-0 items-center gap-[var(--spacing-1)] rounded-full border border-border px-[var(--spacing-4)] py-[var(--spacing-2)] font-mono text-[11px] font-medium whitespace-nowrap text-muted-foreground no-underline"
            >
              <ArrowLeft aria-hidden="true" className="h-3 w-3" />
              Back to the story
            </a>
          </>
        }
      />
    </>
  );
}

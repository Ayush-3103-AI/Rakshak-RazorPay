// The evidence panel's chrome: the shared rail, and the same rail as chips on
// a narrow viewport.
//
// This used to be a bespoke collapsible rail plus a breadcrumb status bar plus
// a theme toggle. All three are gone. The rail is now the one in
// components/Rail.jsx, shared with the story, so both halves of the site
// navigate identically; the breadcrumb duplicated the rail's own active state
// and cost every screen 52px of height it now keeps; and the theme toggle went
// with the light theme, because glass on a light ground is not glass.
//
// The section groups still come from the manifest (sections.js), so adding a
// page cannot desync them.
import { Lock } from "lucide-react";
import Brand from "./components/Brand.jsx";
import Rail, { RailChips } from "./components/Rail.jsx";
import { useArtifact } from "./lib/artifacts.js";
import { SECTIONS } from "./sections.js";
import {
  Activity,
  ArrowLeft,
  Ban,
  FlaskConical,
  GitBranch,
  KeyRound,
  LayoutDashboard,
  Table2,
  TrendingUp,
} from "lucide-react";

const ICONS = {
  verdict: LayoutDashboard,
  lineage: GitBranch,
  ladder: Table2,
  sweep: TrendingUp,
  null: Activity,
  method: KeyRound,
  killed: Ban,
  reproduce: FlaskConical,
};

const ITEMS = SECTIONS.map((s) => ({ ...s, icon: ICONS[s.id] ?? LayoutDashboard }));

/** The way home, plus the one piece of state that governs how every figure
    on this panel must be read. */
function Footer({ opens }) {
  return (
    <div className="flex flex-col gap-[var(--spacing-4)]">
      <div className="glass-flat rounded-[var(--radius-md)] p-[var(--spacing-4)]">
        <p className="m-0 flex items-center gap-[var(--spacing-2)] font-mono text-[10px] font-bold tracking-[0.16em] text-faint uppercase">
          <Lock aria-hidden="true" className="h-3 w-3" />
          test split
        </p>
        <p className="m-0 mt-[var(--spacing-2)] font-mono text-xs text-foreground">
          opened {opens} time{opens === 1 ? "" : "s"}
        </p>
      </div>
      <a
        href="#/"
        className="inline-flex items-center gap-[var(--spacing-3)] rounded-[var(--radius-md)] border border-border px-[var(--spacing-5)] py-[var(--spacing-4)] font-mono text-[11px] font-bold tracking-[0.14em] text-muted-foreground uppercase no-underline transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
      >
        <ArrowLeft aria-hidden="true" className="h-3.5 w-3.5" />
        Back to the story
      </a>
    </div>
  );
}

export default function Chrome({ activeSection, onSelect }) {
  const lockState = useArtifact("lock_state");
  const locks = lockState.data?.payload?.locks ?? [];
  const opens = locks.reduce((n, l) => n + (l.open_count ?? 0), 0);
  const activeId = SECTIONS[activeSection]?.id ?? SECTIONS[0].id;

  return (
    <>
      <Rail
        items={ITEMS}
        activeId={activeId}
        onSelect={onSelect}
        footer={<Footer opens={opens} />}
        label="Panel sections"
      />
      <RailChips
        items={ITEMS}
        activeId={activeId}
        onSelect={onSelect}
        label="Panel sections"
        leading={
          <>
            <Brand href="#/" descriptor={null} className="mr-[var(--spacing-3)] shrink-0" />
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

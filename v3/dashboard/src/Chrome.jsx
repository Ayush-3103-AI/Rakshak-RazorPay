// The shell chrome: a collapsible left rail and a thin status bar.
//
// The rail is grouped rather than flat because eight entries in one list is a
// list you scan; four groups of two is a structure you learn. The groups are
// derived from the manifest (sections.js), so adding a page cannot desync them.
//
// The active tick is a shared element (`layoutId`), so moving between pages
// slides one indicator rather than cross-fading eight. Under reduced motion the
// slide is suppressed and the indicator simply appears — the position, which is
// the information, is identical either way.
//
// Collapsed, the rail keeps every affordance: the icons stay, and the label
// moves into a tooltip rather than disappearing. A rail that hides its
// destinations when narrow is a rail that has to be un-collapsed to be used.
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  Activity,
  ArrowLeft,
  Ban,
  ChevronLeft,
  FlaskConical,
  GitBranch,
  KeyRound,
  LayoutDashboard,
  Lock,
  ShieldCheck,
  Table2,
  TrendingUp,
} from "lucide-react";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { Tooltip } from "./components/ui/Tooltip.jsx";
import { useArtifact } from "./lib/artifacts.js";
import { cn } from "./lib/cn.js";
import { SECTIONS, SECTION_GROUPS } from "./sections.js";

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

export const RAIL_WIDE = 252;
export const RAIL_NARROW = 72;

function RailItem({ section, active, collapsed, onSelect }) {
  const Icon = ICONS[section.id] ?? LayoutDashboard;
  const reduce = useReducedMotion();

  const button = (
    <button
      type="button"
      onClick={() => onSelect(section.id)}
      aria-current={active ? "true" : undefined}
      className={cn(
        "group relative flex w-full cursor-pointer items-center gap-[var(--spacing-4)] rounded-[var(--radius-md)] px-[var(--spacing-4)] py-[var(--spacing-3)] text-left text-sm font-medium transition-colors duration-[var(--duration-quick)]",
        collapsed && "justify-center px-0",
        active ? "text-foreground" : "text-muted-foreground hover:bg-canvas-well hover:text-foreground"
      )}
    >
      {active && (
        <motion.span
          layoutId="rail-active"
          aria-hidden="true"
          className="absolute inset-0 -z-10 rounded-[var(--radius-md)] border border-border-strong bg-linear-to-r from-card to-canvas-well shadow-[var(--shadow-low)]"
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34, mass: 0.7 }}
        />
      )}
      {active && !collapsed && (
        <motion.span
          layoutId="rail-accent"
          aria-hidden="true"
          className="absolute top-[9px] bottom-[9px] left-0 w-[3px] rounded-full bg-primary"
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34, mass: 0.7 }}
        />
      )}
      <Icon
        aria-hidden="true"
        className={cn("h-[18px] w-[18px] shrink-0", active ? "text-primary" : "text-faint group-hover:text-muted-foreground")}
      />
      {!collapsed && (
        <span className="flex min-w-0 flex-1 items-baseline justify-between gap-[var(--spacing-3)]">
          <span className="truncate">{section.label}</span>
          <span className="shrink-0 font-mono text-2xs text-faint">{section.eyebrow}</span>
        </span>
      )}
    </button>
  );

  return collapsed ? (
    <Tooltip content={`${section.eyebrow} · ${section.label}`}>{button}</Tooltip>
  ) : (
    button
  );
}

export default function Chrome({ activeSection, collapsed, onToggleCollapsed, onSelect }) {
  const reduce = useReducedMotion();
  const lockState = useArtifact("lock_state");
  const locks = lockState.data?.payload?.locks ?? [];
  const opens = locks.reduce((n, l) => n + (l.open_count ?? 0), 0);
  const current = SECTIONS[activeSection] ?? SECTIONS[0];

  return (
    <>
      {/* ---- left rail ---------------------------------------------------- */}
      <motion.nav
        aria-label="Panel sections"
        initial={false}
        animate={{ width: collapsed ? RAIL_NARROW : RAIL_WIDE }}
        transition={reduce ? { duration: 0 } : { duration: 0.28, ease: [0, 0, 0.2, 1] }}
        className="glass fixed top-0 left-0 z-30 flex h-screen flex-col gap-[var(--spacing-5)] !rounded-none !border-y-0 !border-l-0 px-[var(--spacing-4)] py-[var(--spacing-6)] max-md:hidden"
      >
        {/* brand + collapse control */}
        <div className={cn("flex items-center gap-[var(--spacing-3)]", collapsed && "justify-center")}>
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-[var(--radius-md)] border border-border bg-card">
            <ShieldCheck aria-hidden="true" className="h-[18px] w-[18px] text-primary" />
          </span>
          <AnimatePresence initial={false}>
            {!collapsed && (
              <motion.span
                key="wordmark"
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: reduce ? 0 : 0.16 }}
                className="flex min-w-0 flex-col"
              >
                <span className="truncate font-heading text-sm font-bold tracking-tight text-foreground">
                  RAKSHAK
                </span>
                <span className="truncate text-2xs text-faint">Merchant risk sentinel</span>
              </motion.span>
            )}
          </AnimatePresence>
          {!collapsed && (
            <button
              type="button"
              onClick={onToggleCollapsed}
              aria-label="Collapse sidebar"
              aria-expanded="true"
              className="ml-auto grid h-8 w-8 shrink-0 cursor-pointer place-items-center rounded-[var(--radius-sm)] border border-border bg-card text-faint transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
            >
              <ChevronLeft aria-hidden="true" className="h-4 w-4" />
            </button>
          )}
        </div>

        {collapsed && (
          <button
            type="button"
            onClick={onToggleCollapsed}
            aria-label="Expand sidebar"
            aria-expanded="false"
            className="mx-auto grid h-8 w-8 cursor-pointer place-items-center rounded-[var(--radius-sm)] border border-border bg-card text-faint transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
          >
            <ChevronLeft aria-hidden="true" className="h-4 w-4 rotate-180" />
          </button>
        )}

        {/* the way back to the front door */}
        {collapsed ? (
          <Tooltip content="Back to the story">
            <a
              href="#/"
              aria-label="Back to the story"
              className="mx-auto grid h-8 w-8 place-items-center rounded-[var(--radius-sm)] border border-border bg-card text-faint transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
            >
              <ArrowLeft aria-hidden="true" className="h-4 w-4" />
            </a>
          </Tooltip>
        ) : (
          <a
            href="#/"
            className="inline-flex items-center gap-[var(--spacing-2)] rounded-[var(--radius-sm)] border border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-2)] text-xs font-medium text-muted-foreground no-underline transition-colors duration-[var(--duration-quick)] hover:border-border-strong hover:text-foreground"
          >
            <ArrowLeft aria-hidden="true" className="h-3.5 w-3.5" />
            Back to the story
          </a>
        )}

        {/* grouped destinations */}
        <div className="flex min-h-0 flex-1 flex-col gap-[var(--spacing-5)] overflow-y-auto">
          {SECTION_GROUPS.map((group) => (
            <div key={group.name} className="flex flex-col gap-[var(--spacing-1)]">
              {collapsed ? (
                <span aria-hidden="true" className="mx-auto my-[var(--spacing-2)] h-px w-5 bg-border" />
              ) : (
                <p className="m-0 px-[var(--spacing-4)] pb-[var(--spacing-2)] text-2xs font-medium tracking-wide text-faint">
                  {group.name}
                </p>
              )}
              {group.items.map((section) => (
                <RailItem
                  key={section.id}
                  section={section}
                  active={section.index === activeSection}
                  collapsed={collapsed}
                  onSelect={onSelect}
                />
              ))}
            </div>
          ))}
        </div>

        {/* the one piece of state that governs how every figure must be read */}
        <div className={cn("shrink-0", collapsed && "flex justify-center")}>
          {collapsed ? (
            <Tooltip content={`Test split opened ${opens} times`}>
              <span className="grid h-8 w-8 place-items-center rounded-[var(--radius-sm)] border border-border bg-card">
                <Lock aria-hidden="true" className="h-3.5 w-3.5 text-faint" />
              </span>
            </Tooltip>
          ) : (
            <div className="rounded-[var(--radius-md)] border border-border bg-card p-[var(--spacing-4)]">
              <p className="m-0 flex items-center gap-[var(--spacing-2)] font-mono text-2xs font-bold tracking-wide text-faint uppercase">
                <Lock aria-hidden="true" className="h-3 w-3" />
                test split
              </p>
              <p className="m-0 mt-[var(--spacing-2)] font-mono text-xs text-foreground">
                opened {opens} time{opens === 1 ? "" : "s"}
              </p>
            </div>
          )}
        </div>
      </motion.nav>

      {/* ---- status bar ---------------------------------------------------- */}
      <header
        className="fixed top-0 right-0 z-20 flex h-[52px] items-center justify-between gap-[var(--spacing-4)] border-b border-border bg-background/85 px-[var(--spacing-7)] backdrop-blur-md max-md:left-0 max-md:px-[var(--spacing-4)]"
        style={{ left: collapsed ? RAIL_NARROW : RAIL_WIDE }}
      >
        <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-[var(--spacing-3)] text-sm">
          <span className="shrink-0 text-faint max-md:hidden">{current.group}</span>
          <span aria-hidden="true" className="shrink-0 text-faint max-md:hidden">
            /
          </span>
          <span className="truncate font-medium text-foreground">{current.label}</span>
          <span className="shrink-0 font-mono text-2xs text-faint">{current.eyebrow}</span>
        </nav>
        <div className="flex shrink-0 items-center gap-[var(--spacing-4)]">
          <span className="font-mono text-2xs text-faint max-md:hidden">
            {SECTIONS.length} screens · ~2 min
          </span>
          <ThemeToggle />
        </div>
      </header>

      {/* ---- mobile: the rail becomes a scrolling chip strip --------------- */}
      <nav
        aria-label="Panel sections"
        className="fixed top-[52px] right-0 left-0 z-20 hidden gap-[var(--spacing-2)] overflow-x-auto border-b border-border bg-background/95 px-[var(--spacing-4)] py-[var(--spacing-3)] backdrop-blur-md max-md:flex"
      >
        <a
          href="#/"
          className="inline-flex shrink-0 items-center gap-[var(--spacing-1)] rounded-full border border-border px-[var(--spacing-4)] py-[var(--spacing-2)] text-xs font-medium whitespace-nowrap text-muted-foreground no-underline"
        >
          <ArrowLeft aria-hidden="true" className="h-3 w-3" />
          Story
        </a>
        {SECTIONS.map((section, i) => (
          <button
            key={section.id}
            type="button"
            onClick={() => onSelect(section.id)}
            aria-current={i === activeSection ? "true" : undefined}
            className={cn(
              "shrink-0 cursor-pointer rounded-full border px-[var(--spacing-4)] py-[var(--spacing-2)] text-xs font-medium whitespace-nowrap transition-colors duration-[var(--duration-quick)]",
              i === activeSection
                ? "border-border-strong bg-card text-foreground"
                : "border-transparent text-muted-foreground"
            )}
          >
            {section.label}
          </button>
        ))}
      </nav>
    </>
  );
}

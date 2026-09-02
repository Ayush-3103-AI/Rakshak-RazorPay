// Persistent left rail: section nav (left-aligned and grid-anchored, per a
// dashboard rather than a centered marketing page) plus the theme toggle.
//
// The active tick is a shared element (`layoutId`), so moving between sections
// slides one indicator rather than cross-fading six. Under reduced motion the
// slide is suppressed and the indicator simply appears — the position, which is
// the information, is identical either way.
import { motion, useReducedMotion } from "framer-motion";
import { Ban, FlaskConical, GitBranch, LayoutDashboard, ShieldCheck, Table2 } from "lucide-react";
import { SECTIONS } from "./sections.js";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { cn } from "./lib/cn.js";

const ICONS = {
  verdict: LayoutDashboard,
  generations: GitBranch,
  method: ShieldCheck,
  results: Table2,
  killed: Ban,
  reproduce: FlaskConical,
};

export default function Chrome({ activeSection }) {
  const reduce = useReducedMotion();
  const go = (id) => document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" });

  return (
    <nav
      aria-label="Sections"
      className="fixed top-0 left-0 z-20 flex h-screen w-[220px] flex-col gap-[var(--spacing-1)] border-r border-border bg-card px-[var(--spacing-5)] py-[var(--spacing-7)] max-md:relative max-md:h-auto max-md:w-full max-md:flex-row max-md:items-center max-md:gap-[var(--spacing-4)] max-md:overflow-x-auto max-md:border-r-0 max-md:border-b max-md:px-[var(--spacing-4)] max-md:py-[var(--spacing-3)]"
    >
      <div className="mb-[var(--spacing-2)] max-md:hidden">
        <span className="font-mono text-xs font-bold tracking-[0.2em] text-foreground">RAKSHAK</span>
        <p className="m-0 mt-[2px] text-2xs text-faint">G3 evidence panel</p>
      </div>

      {SECTIONS.map((s, i) => {
        const Icon = ICONS[s.id];
        const active = i === activeSection;
        return (
          <button
            key={s.id}
            type="button"
            onClick={() => go(s.id)}
            aria-current={active ? "true" : undefined}
            className={cn(
              "group relative flex cursor-pointer items-center gap-[var(--spacing-3)] rounded-[var(--radius-sm)] border border-transparent px-[var(--spacing-3)] py-[var(--spacing-3)] text-left text-sm font-medium transition-colors duration-[var(--duration-quick)] max-md:whitespace-nowrap",
              active ? "text-foreground" : "text-muted-foreground hover:bg-canvas-well hover:text-foreground"
            )}
          >
            {active && (
              <motion.span
                layoutId="rail-active"
                aria-hidden="true"
                className="absolute inset-0 -z-10 rounded-[var(--radius-sm)] border border-border bg-canvas-well"
                transition={
                  reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34, mass: 0.7 }
                }
              />
            )}
            <Icon
              aria-hidden="true"
              className={cn("h-4 w-4 shrink-0", active ? "text-primary" : "text-faint group-hover:text-muted-foreground")}
            />
            <span className="flex flex-col">
              <span className="font-mono text-2xs tracking-wide text-faint">{s.eyebrow}</span>
              <span>{s.label}</span>
            </span>
          </button>
        );
      })}

      <div className="mt-auto pt-[var(--spacing-5)] max-md:mt-0 max-md:pt-0">
        <ThemeToggle />
      </div>
    </nav>
  );
}

// One rail, both halves of the site, on the RIGHT.
//
// It sits on the right because the story is read left to right: the headline
// owns the left edge and the type can run to its natural measure without a
// navigation column pushing it inward. The panel uses the same side so the two
// halves stay one product.
//
// THE SPINE IS THE JOURNEY. A hairline runs down the gutter behind the
// markers, lit from the top to the current entry, so progress through the
// sequence is legible without reading a single label — "you are on four of
// eight", which the hover-only dots this replaces could never say.
//
// It is deliberately quiet. Every label is 12px at medium weight, the group
// titles are 9px and dim, and the markers are small: this is a map you consult,
// not a second headline competing with the one on screen. Quiet is not hidden,
// though — nothing here waits for a hover to become readable, which was the
// actual failing of the dots.
import { motion, useReducedMotion } from "framer-motion";
import Brand from "./Brand.jsx";
import { cn } from "../lib/cn.js";

export const RAIL_WIDTH = 214;

function pad(n) {
  return String(n + 1).padStart(2, "0");
}

function RailItem({ item, index, state, onSelect }) {
  const reduce = useReducedMotion();
  const active = state === "active";
  const done = state === "done";

  return (
    <li>
      <button
        type="button"
        onClick={() => onSelect(item.id)}
        aria-current={active ? "true" : undefined}
        className={cn(
          "group relative flex w-full cursor-pointer items-center gap-[var(--spacing-3)] rounded-[var(--radius-sm)] py-[6px] pr-[var(--spacing-3)] pl-[var(--spacing-2)] text-left transition-colors duration-[var(--duration-quick)]",
          active ? "text-foreground" : "text-muted-foreground/85 hover:text-foreground"
        )}
      >
        {active && (
          <motion.span
            layoutId="rail-active"
            aria-hidden="true"
            className="absolute inset-0 -z-10 rounded-[var(--radius-sm)] bg-primary/10"
            transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34, mass: 0.7 }}
          />
        )}

        {/* the marker, sitting on the spine */}
        <span aria-hidden="true" className="relative grid h-[14px] w-[14px] shrink-0 place-items-center">
          <span
            className={cn(
              "block rounded-full transition-all duration-[var(--duration-moderate)]",
              active
                ? "h-[8px] w-[8px] bg-primary shadow-[0_0_10px_1px_var(--color-primary)]"
                : done
                  ? "h-[5px] w-[5px] bg-primary/60"
                  : "h-[5px] w-[5px] bg-border-strong group-hover:bg-faint"
            )}
          />
        </span>

        <span
          className={cn(
            "shrink-0 font-mono text-[10px] tabular-nums transition-colors duration-[var(--duration-quick)]",
            active ? "text-primary-text" : "text-faint/80"
          )}
        >
          {item.eyebrow ?? pad(index)}
        </span>

        <span className={cn("min-w-0 flex-1 truncate text-xs", active ? "font-semibold" : "font-medium")}>
          {item.label}
        </span>
      </button>
    </li>
  );
}

/**
 * @param items    [{ id, label, eyebrow?, group? }] in sequence order
 * @param activeId the item currently on screen
 * @param onSelect (id) => void
 * @param footer   anything to park at the bottom (a stat, a route switch)
 * @param label    the nav's accessible name
 */
export default function Rail({ items, activeId, onSelect, footer, label = "Sections" }) {
  const activeIndex = Math.max(0, items.findIndex((i) => i.id === activeId));

  // Grouped only if the items say so, so the story's three groups and the
  // panel's four are the same component with the same markup.
  const groups = [];
  items.forEach((item, index) => {
    const name = item.group ?? null;
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.items.push({ item, index });
    else groups.push({ name, items: [{ item, index }] });
  });

  return (
    <nav
      aria-label={label}
      style={{ width: RAIL_WIDTH }}
      className="glass fixed top-0 right-0 z-30 flex h-screen flex-col gap-[var(--spacing-6)] !rounded-none !border-y-0 !border-r-0 px-[var(--spacing-4)] py-[var(--spacing-6)] max-lg:hidden"
    >
      <Brand href="#/" size="xs" className="px-[var(--spacing-2)]" />

      <div className="relative flex min-h-0 flex-1 flex-col gap-[var(--spacing-5)] overflow-y-auto">
        {/* the spine, and the lit portion of it — the journey, as one line */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute top-[14px] bottom-[14px] left-[15px] w-px bg-border"
        />
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute top-[14px] bottom-[14px] left-[15px] w-px origin-top bg-primary/60"
          initial={false}
          animate={{ scaleY: items.length > 1 ? activeIndex / (items.length - 1) : 1 }}
          transition={{ duration: 0.45, ease: [0, 0, 0.2, 1] }}
        />

        {groups.map((group, gi) => (
          <div key={group.name ?? gi} className="flex flex-col gap-[2px]">
            {group.name && (
              <p className="m-0 pb-[var(--spacing-2)] pl-[var(--spacing-2)] font-mono text-[9px] font-medium tracking-[0.2em] text-faint/60 uppercase">
                {group.name}
              </p>
            )}
            <ul className="m-0 flex list-none flex-col gap-[2px] p-0">
              {group.items.map(({ item, index }) => (
                <RailItem
                  key={item.id}
                  item={item}
                  index={index}
                  state={index === activeIndex ? "active" : index < activeIndex ? "done" : "todo"}
                  onSelect={onSelect}
                />
              ))}
            </ul>
          </div>
        ))}
      </div>

      {footer && <div className="shrink-0">{footer}</div>}
    </nav>
  );
}

/** The same destinations as a scrolling chip strip, for narrow viewports. */
export function RailChips({ items, activeId, onSelect, leading, label = "Sections" }) {
  return (
    <nav
      aria-label={label}
      className="glass fixed top-0 right-0 left-0 z-30 hidden items-center gap-[var(--spacing-2)] overflow-x-auto !rounded-none !border-x-0 !border-t-0 px-[var(--spacing-4)] py-[var(--spacing-3)] max-lg:flex"
    >
      {leading}
      {items.map((item, i) => (
        <button
          key={item.id}
          type="button"
          onClick={() => onSelect(item.id)}
          aria-current={item.id === activeId ? "true" : undefined}
          className={cn(
            "shrink-0 cursor-pointer rounded-full border px-[var(--spacing-4)] py-[var(--spacing-2)] font-mono text-[11px] font-medium tracking-[0.08em] whitespace-nowrap transition-colors duration-[var(--duration-quick)]",
            item.id === activeId
              ? "border-primary/40 bg-primary/15 text-primary-text"
              : "border-border text-muted-foreground"
          )}
        >
          <span className="mr-[var(--spacing-2)] text-faint">{item.eyebrow ?? pad(i)}</span>
          {item.label}
        </button>
      ))}
    </nav>
  );
}

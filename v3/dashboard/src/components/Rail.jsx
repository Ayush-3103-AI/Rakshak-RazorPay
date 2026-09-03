// The page list, on the right, with no panel around it.
//
// Borderless on purpose: it is a marker of where you are, not a piece of
// furniture. No glass, no border, no background — just the page titles, their
// numbers, and the spine. Anything that belongs to the site rather than to the
// sequence (the brand, the route switch, the lock count) lives in the top bar,
// which is where it was before this got promoted into a full sidebar.
//
// THE SPINE IS THE JOURNEY. A hairline behind the markers, lit from the top
// down to the current page, so how far in you are is legible without reading a
// word. It animates between pages.
//
// Every title is readable at rest. The version this restores revealed labels
// only on hover, which is invisible to anyone who never hovers and to everyone
// on a touchscreen; quiet is the goal, hidden is not.
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "../lib/cn.js";

// What the main column must leave clear on the right. The list itself is
// narrower; this is the list plus its breathing room.
export const RAIL_WIDTH = 208;

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
          "group flex w-full cursor-pointer items-center gap-[var(--spacing-3)] rounded-[var(--radius-xs)] py-[5px] text-left transition-colors duration-[var(--duration-quick)]",
          active ? "text-foreground" : "text-muted-foreground/70 hover:text-foreground"
        )}
      >
        <span aria-hidden="true" className="relative grid h-[14px] w-[14px] shrink-0 place-items-center">
          <motion.span
            layout={!reduce}
            className={cn(
              "block rounded-full transition-colors duration-[var(--duration-moderate)]",
              active
                ? "h-[8px] w-[8px] bg-primary shadow-[0_0_10px_1px_var(--color-primary)]"
                : done
                  ? "h-[5px] w-[5px] bg-primary/55"
                  : "h-[5px] w-[5px] bg-border-strong group-hover:bg-faint"
            )}
          />
        </span>

        <span
          className={cn(
            "shrink-0 font-mono text-[10px] tabular-nums transition-colors duration-[var(--duration-quick)]",
            active ? "text-primary-text" : "text-faint/70"
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
 * @param items    [{ id, label, eyebrow? }] in sequence order
 * @param activeId the page currently on screen
 * @param onSelect (id) => void
 * @param label    the nav's accessible name
 */
export default function Rail({ items, activeId, onSelect, label = "Pages" }) {
  const activeIndex = Math.max(0, items.findIndex((i) => i.id === activeId));

  return (
    <nav
      aria-label={label}
      className="fixed top-1/2 right-[clamp(12px,2vw,28px)] z-30 -translate-y-1/2 max-lg:hidden"
    >
      <div className="relative">
        {/* the spine, and the lit portion of it */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute top-[12px] bottom-[12px] left-[7px] w-px bg-border"
        />
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute top-[12px] bottom-[12px] left-[7px] w-px origin-top bg-primary/60"
          initial={false}
          animate={{ scaleY: items.length > 1 ? activeIndex / (items.length - 1) : 1 }}
          transition={{ duration: 0.45, ease: [0, 0, 0.2, 1] }}
        />

        <ul className="m-0 flex list-none flex-col gap-0 p-0">
          {items.map((item, index) => (
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
    </nav>
  );
}

/** The same destinations as a scrolling chip strip, for narrow viewports. */
export function RailChips({ items, activeId, onSelect, leading, label = "Pages" }) {
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

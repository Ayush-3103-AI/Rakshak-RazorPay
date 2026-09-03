// One rail, both halves of the site.
//
// The story and the evidence panel are each a fixed sequence of full screens,
// so they want the same navigation object: the whole sequence visible at once,
// every entry reachable in one click, and an unmistakable answer to "where am
// I and how much is left". This renders that, and both routes pass it their
// own items.
//
// THE SPINE IS THE POINT. A vertical line runs down the gutter behind the
// markers, lit from the top to the current entry, so progress through the
// sequence is legible without reading a single label. The dots alone said
// "there are eight things"; the lit spine says "you are on four of eight",
// which is the question a reader hunting a two-minute read is actually asking.
//
// The rail does not collapse. A rail that has to be un-collapsed to be used is
// a rail that is not doing its job, and the previous build hid its labels
// until hover — which is invisible to anyone who never hovers.
import { motion, useReducedMotion } from "framer-motion";
import Brand from "./Brand.jsx";
import { cn } from "../lib/cn.js";

export const RAIL_WIDTH = 268;

function pad(n) {
  return String(n + 1).padStart(2, "0");
}

function RailItem({ item, index, state, onSelect }) {
  const reduce = useReducedMotion();
  const Icon = item.icon;
  const active = state === "active";
  const done = state === "done";

  return (
    <li className="relative">
      <button
        type="button"
        onClick={() => onSelect(item.id)}
        aria-current={active ? "true" : undefined}
        className={cn(
          "group relative flex w-full cursor-pointer items-center gap-[var(--spacing-4)] rounded-[var(--radius-md)] py-[var(--spacing-3)] pr-[var(--spacing-4)] pl-[var(--spacing-3)] text-left transition-colors duration-[var(--duration-quick)]",
          active ? "text-foreground" : "text-muted-foreground hover:text-foreground"
        )}
      >
        {active && (
          <motion.span
            layoutId="rail-active"
            aria-hidden="true"
            className="absolute inset-0 -z-10 rounded-[var(--radius-md)] border border-primary/35 bg-primary/12"
            transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 420, damping: 34, mass: 0.7 }}
          />
        )}

        {/* the marker, sitting on the spine */}
        <span aria-hidden="true" className="relative grid h-[18px] w-[18px] shrink-0 place-items-center">
          <span
            className={cn(
              "block rounded-full transition-all duration-[var(--duration-moderate)]",
              active
                ? "h-[10px] w-[10px] bg-primary shadow-[0_0_12px_2px_var(--color-primary)]"
                : done
                  ? "h-[7px] w-[7px] bg-primary/70"
                  : "h-[7px] w-[7px] bg-border-strong group-hover:bg-faint"
            )}
          />
        </span>

        <span
          className={cn(
            "shrink-0 font-mono text-[11px] tabular-nums transition-colors duration-[var(--duration-quick)]",
            active ? "text-primary-text" : "text-faint"
          )}
        >
          {item.eyebrow ?? pad(index)}
        </span>

        <span className={cn("min-w-0 flex-1 truncate text-sm", active ? "font-semibold" : "font-medium")}>
          {item.label}
        </span>

        {Icon && (
          <Icon
            aria-hidden="true"
            className={cn(
              "h-[15px] w-[15px] shrink-0 transition-colors duration-[var(--duration-quick)]",
              active ? "text-primary" : "text-faint/70 group-hover:text-faint"
            )}
          />
        )}
      </button>
    </li>
  );
}

/**
 * @param items    [{ id, label, eyebrow?, icon?, group? }] in sequence order
 * @param activeId the item currently on screen
 * @param onSelect (id) => void
 * @param footer   anything to park at the bottom (a stat, a route switch)
 * @param label    the nav's accessible name
 */
export default function Rail({ items, activeId, onSelect, footer, label = "Sections" }) {
  const activeIndex = Math.max(0, items.findIndex((i) => i.id === activeId));

  // Grouped only if the items say so, so the story's flat list and the panel's
  // four groups are the same component with the same markup.
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
      className="glass fixed top-0 left-0 z-30 flex h-screen flex-col gap-[var(--spacing-6)] !rounded-none !border-y-0 !border-l-0 px-[var(--spacing-5)] py-[var(--spacing-7)] max-lg:hidden"
    >
      <Brand href="#/" />

      <div className="relative flex min-h-0 flex-1 flex-col gap-[var(--spacing-6)] overflow-y-auto">
        {/* the spine, and the lit portion of it */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute top-[18px] bottom-[18px] left-[20px] w-px bg-border"
        />
        <motion.span
          aria-hidden="true"
          className="pointer-events-none absolute top-[18px] left-[20px] w-px origin-top bg-primary/70"
          initial={false}
          animate={{ scaleY: items.length > 1 ? activeIndex / (items.length - 1) : 1 }}
          transition={{ duration: 0.4, ease: [0, 0, 0.2, 1] }}
          style={{ bottom: 18 }}
        />

        {groups.map((group, gi) => (
          <div key={group.name ?? gi} className="flex flex-col gap-[var(--spacing-1)]">
            {group.name && (
              <p className="m-0 pb-[var(--spacing-2)] pl-[var(--spacing-3)] font-mono text-[10px] font-bold tracking-[0.18em] text-faint uppercase">
                {group.name}
              </p>
            )}
            <ul className="m-0 flex list-none flex-col gap-[var(--spacing-1)] p-0">
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

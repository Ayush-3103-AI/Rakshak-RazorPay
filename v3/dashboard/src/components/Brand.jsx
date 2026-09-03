// The Rakshak mark.
//
// Rakshak (रक्षक) is "protector". The mark is a shield with a monitoring trace
// running through it — a flat baseline, one anomalous spike, and the return to
// baseline. That is literally what this system does: watch a merchant's daily
// signal, and act on the day it departs from its own normal. The shield says
// what it is for; the trace says how it works.
//
// Drawn on a 24-unit grid with 2-unit strokes so it survives at 18px in a rail
// and at 96px on a title screen. Two weights of the brand colour separate the
// vessel from the signal, and the spike is the only thing in the brighter one,
// so the eye lands on the anomaly — which is the product in one glyph.
import { cn } from "../lib/cn.js";

export function Mark({ className, title }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      role={title ? "img" : "presentation"}
      aria-label={title}
      aria-hidden={title ? undefined : "true"}
      className={cn("h-full w-full", className)}
    >
      {/* the shield */}
      <path
        d="M12 2.4 20.4 5.4V11.7C20.4 16.9 17 20.6 12 21.9 7 20.6 3.6 16.9 3.6 11.7V5.4Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
        opacity="0.55"
      />
      {/* the trace: baseline, the departure, and back */}
      <path
        d="M7 13.4h2.1l1.5-4.2 1.9 6.1 1.2-2.9H17"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const TILE = {
  sm: "h-9 w-9 rounded-[var(--radius-md)] p-[7px]",
  md: "h-11 w-11 rounded-[var(--radius-lg)] p-[9px]",
};

/** The mark in its glass tile — the lockup's constant element. */
export function MarkTile({ size = "sm", className }) {
  return (
    <span
      className={cn(
        "relative grid shrink-0 place-items-center border border-primary/35 bg-primary/12 text-primary-text",
        "shadow-[0_0_24px_-6px_var(--color-primary)]",
        TILE[size],
        className
      )}
    >
      <Mark />
    </span>
  );
}

/**
 * Mark plus wordmark. `descriptor` is the line under the name; pass null for
 * the bare lockup. As a link when `href` is given, so the brand is always the
 * way home.
 */
export default function Brand({
  size = "sm",
  descriptor = "Merchant risk sentinel",
  href,
  className,
}) {
  const Tag = href ? "a" : "div";
  return (
    <Tag
      href={href}
      className={cn("flex min-w-0 items-center gap-[var(--spacing-4)] no-underline", className)}
    >
      <MarkTile size={size} />
      <span className="flex min-w-0 flex-col">
        <span
          className={cn(
            "truncate font-heading font-extrabold text-foreground",
            size === "md" ? "text-lg tracking-[-0.02em]" : "text-base tracking-[-0.015em]"
          )}
        >
          RAKSHAK
        </span>
        {descriptor && (
          <span className="truncate font-mono text-[10px] tracking-[0.16em] text-faint uppercase">
            {descriptor}
          </span>
        )}
      </span>
    </Tag>
  );
}

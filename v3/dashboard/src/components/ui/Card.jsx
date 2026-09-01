import { cn } from "../../lib/cn.js";

// Real product surfaces vary density deliberately — a spacious header card
// reads differently from a dense metric tile. `pad` picks the level rather
// than every card sharing one padding constant.
const PAD = {
  compact: "p-[var(--spacing-5)]",
  regular: "p-[var(--spacing-6)]",
  spacious: "p-[var(--spacing-8)]",
};

const ELEV = {
  low: "shadow-[var(--shadow-low)]",
  mid: "shadow-[var(--shadow-mid)]",
  high: "shadow-[var(--shadow-high)]",
};

export default function Card({ as: Tag = "div", pad = "regular", elevation = "low", className, ...props }) {
  return (
    <Tag
      className={cn(
        "rounded-[var(--radius-lg)] border border-border bg-card text-card-foreground",
        PAD[pad],
        ELEV[elevation],
        className
      )}
      {...props}
    />
  );
}

// The split label every numeric figure on this panel must carry (T-0126's
// contract, #63's AC): "the site must never render an unlocked number in the
// visual register of a final one." VALIDATION gets a dashed border and a
// flask icon — visibly provisional — while TEST gets a solid border and a
// lock. The shape difference matters as much as the color: this label must
// still read correctly in grayscale, on a projector, to someone colorblind.
import { CircleSlash, Database, FlaskConical, Lock } from "lucide-react";
import { cn } from "../../lib/cn.js";

const SPLIT = {
  TRAIN: {
    Icon: Database,
    classes: "text-faint bg-canvas-well border-border",
  },
  VALIDATION: {
    Icon: FlaskConical,
    classes: "text-notice bg-notice-bg border-notice-border border-dashed",
  },
  TEST: {
    Icon: Lock,
    classes: "text-positive bg-positive-bg border-positive-border",
  },
  NULL_RUN: {
    Icon: CircleSlash,
    classes: "text-information bg-information-bg border-information-border",
  },
};

export default function SplitChip({ split, className }) {
  if (!split) return null;
  const entry = SPLIT[split] ?? { Icon: FlaskConical, classes: "text-faint bg-canvas-well border-border" };
  const { Icon, classes } = entry;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-[var(--spacing-1)] rounded-[var(--radius-xs)] border px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold tracking-widest uppercase",
        classes,
        className
      )}
      title={
        split === "VALIDATION"
          ? "Not yet locked — the test split opens once, in T-0116"
          : split === "TEST"
            ? "Locked test-split figure"
            : split === "NULL_RUN"
              ? "Zero-prevalence synthetic run — belongs to no split"
              : undefined
      }
    >
      <Icon aria-hidden="true" className="h-3 w-3 shrink-0" />
      {split}
    </span>
  );
}

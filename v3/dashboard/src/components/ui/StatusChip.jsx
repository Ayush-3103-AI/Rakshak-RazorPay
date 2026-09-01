// One vocabulary for every status this panel renders — manifest PRESENT/
// MISSING, G5 GREEN/RED, the roster's seven statuses. Every entry pairs a
// color with an icon AND a literal word: color is reinforcement, never the
// only signal (a judge on a projector, or reading in grayscale, must still
// be able to tell PRESENT from MISSING).
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  CircleDashed,
  Clock,
  HelpCircle,
  Package,
  XCircle,
} from "lucide-react";
import { cn } from "../../lib/cn.js";

const TONE = {
  positive: "text-positive bg-positive-bg border-positive-border",
  negative: "text-negative bg-negative-bg border-negative-border",
  notice: "text-notice bg-notice-bg border-notice-border",
  information: "text-information bg-information-bg border-information-border",
  muted: "text-faint bg-canvas-well border-border",
};

// status token -> { label, tone, Icon }. Unlisted tokens still render (as
// their raw string, muted) rather than throwing — an unrecognised roster
// status is a contract bug worth seeing on the page, not a crash.
const REGISTRY = {
  PRESENT: { label: "PRESENT", tone: "positive", Icon: CheckCircle2 },
  MISSING: { label: "MISSING", tone: "notice", Icon: AlertTriangle },
  GREEN: { label: "GREEN", tone: "positive", Icon: CheckCircle2 },
  RED: { label: "RED", tone: "negative", Icon: XCircle },
  planned: { label: "planned", tone: "muted", Icon: CircleDashed },
  built: { label: "built", tone: "information", Icon: Package },
  scored: { label: "scored", tone: "positive", Icon: CheckCircle2 },
  cut: { label: "cut", tone: "negative", Icon: Ban },
  deferred: { label: "deferred", tone: "notice", Icon: Clock },
  conditional: { label: "conditional", tone: "information", Icon: HelpCircle },
  UNVERIFIED: { label: "UNVERIFIED", tone: "notice", Icon: AlertTriangle },
};

export default function StatusChip({ status, className }) {
  const entry = REGISTRY[status] ?? { label: String(status), tone: "muted", Icon: HelpCircle };
  const { label, tone, Icon } = entry;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-[var(--spacing-1)] rounded-full border px-[var(--spacing-3)] py-[var(--spacing-1)] font-mono text-2xs font-bold tracking-wide uppercase",
        TONE[tone],
        className
      )}
    >
      <Icon aria-hidden="true" className="h-3 w-3 shrink-0" />
      {label}
    </span>
  );
}

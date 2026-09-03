// Every policy on the ladder as one bar of savings, against the tick of the
// floor it had to beat. The floor is `volume_rank` — rank merchants by size,
// learn nothing — and one bar crosses it. Rows are sorted by margin so the
// reader's eye lands on the winner without hunting.
//
// A row's floor is ITS OWN floor (`savings_floor_volume_rank` on that row),
// drawn as a tick per row rather than one shared line: two rungs were scored
// on a different sub-population with a different floor, and a shared line
// would misplace them.
import { motion, useReducedMotion } from "framer-motion";
import { fmtSigned } from "../../lib/format.js";

const LABEL_W = 250;
const BAR_X0 = LABEL_W + 20;
const BAR_X1 = 880;
const ROW_H = 30;

export default function MarginBars({ rows }) {
  const reduce = useReducedMotion();
  const usable = rows.filter((r) => r.savings != null);
  if (!usable.length) return null;

  const max = Math.max(...usable.flatMap((r) => [r.savings, r.floor ?? 0])) * 1.06;
  const scale = (v) => BAR_X0 + (Math.max(0, v) / max) * (BAR_X1 - BAR_X0);
  const height = usable.length * ROW_H + 36;

  return (
    <svg
      viewBox={`0 0 1000 ${height}`}
      role="img"
      aria-label={`Savings per policy against the volume_rank floor. ${usable.filter((r) => r.beats).map((r) => r.label).join(", ") || "No policy"} beats every floor.`}
      className="block h-auto w-full"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {usable.map((r, i) => {
        const y = 18 + i * ROW_H + ROW_H / 2;
        // Ahead on the mean but not on every seed is the seed-flipping case the
        // README warns about; it gets the notice colour, never the winner's.
        const tone = r.beats ? "var(--color-primary)" : r.margin != null && r.margin > 0 ? "var(--color-notice)" : "var(--color-faint)";
        return (
          <g key={r.label}>
            <text
              x={LABEL_W}
              y={y + 4}
              textAnchor="end"
              fontSize="12"
              fontWeight={r.beats ? 700 : 500}
              fill={r.beats ? "var(--color-foreground)" : "var(--color-muted-foreground)"}
            >
              {r.label}
            </text>
            {/* track */}
            <path d={`M${BAR_X0} ${y}H${BAR_X1}`} stroke="var(--color-border)" strokeWidth="1" />
            {/* the bar */}
            <motion.path
              d={`M${BAR_X0} ${y}H${scale(r.savings)}`}
              stroke={tone}
              strokeWidth={r.beats ? 14 : 10}
              strokeLinecap="round"
              initial={reduce ? false : { pathLength: 0, opacity: 0 }}
              whileInView={{ pathLength: 1, opacity: r.beats ? 1 : 0.55 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.9, delay: reduce ? 0 : 0.05 * i, ease: [0.2, 0.65, 0.2, 1] }}
            />
            {/* the floor tick */}
            {r.floor != null && (
              <path
                d={`M${scale(r.floor)} ${y - 11}V${y + 11}`}
                stroke="var(--color-foreground)"
                strokeWidth="2"
                strokeLinecap="round"
                opacity={0.85}
              />
            )}
            <text
              x={BAR_X1 + 16}
              y={y + 4}
              fontSize="12"
              fontWeight={r.beats ? 700 : 500}
              fill={r.beats ? "var(--color-primary-text)" : "var(--color-faint)"}
            >
              {r.margin != null ? fmtSigned(r.margin, 3) : "—"}
            </text>
          </g>
        );
      })}
      <text x={BAR_X1 + 16} y={12} fontSize="10" fill="var(--color-faint)" letterSpacing="0.12em">
        MARGIN
      </text>
      <text x={BAR_X0} y={12} fontSize="10" fill="var(--color-faint)" letterSpacing="0.12em">
        SAVINGS · TICK = THE FLOOR TO BEAT
      </text>
    </svg>
  );
}

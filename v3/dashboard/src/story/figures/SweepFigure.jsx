// The cost sweep, drawn by the scrollbar: the survivor's savings across the
// swept cost ratios, the floor as a dashed line under it, the other policies
// as faint lines behind. The survivor's line draws last, left to right, so
// the reader watches it stay above the floor at every ratio.
//
// Log x-axis — the ratios are the artifact's grid, four orders of magnitude.
// Every value on the plot is read from the sweep's own series; the y-domain
// is fitted to them with padding, never fixed.
import { motion, useTransform } from "framer-motion";
import { fmtNum } from "../../lib/format.js";

const PX = 100;
const PX1 = 930;
const PY = 44;
const PY1 = 370;

export default function SweepFigure({ progress, sweep }) {
  const backdrop = useTransform(progress, [0, 0.2], [0, 1]);
  const others = useTransform(progress, [0.1, 0.35], [0, 1]);
  const floor = useTransform(progress, [0.25, 0.5], [0, 1]);
  const hero = useTransform(progress, [0.45, 0.85], [0, 1]);
  const points = useTransform(progress, [0.82, 0.96], [0, 1]);
  const marker = useTransform(progress, [0.6, 0.75], [0, 1]);

  const { ratios, series, floor: floorSeries, others: otherSeries, shippedRatio, shippedWithin, band, policy } = sweep;
  if (!ratios.length || !series) return null;

  const logs = ratios.map((r) => Math.log10(r));
  const lx0 = Math.min(...logs);
  const lx1 = Math.max(...logs);
  const x = (r) => PX + ((Math.log10(r) - lx0) / (lx1 - lx0 || 1)) * (PX1 - PX);

  const all = [...series.values, ...(floorSeries?.values ?? []), ...otherSeries.flatMap((s) => s.values)].filter(Number.isFinite);
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const pad = (hi - lo || 0.1) * 0.18;
  const y0 = lo - pad;
  const y1 = hi + pad;
  const y = (v) => PY1 - ((v - y0) / (y1 - y0)) * (PY1 - PY);

  const line = (values) => values.map((v, i) => `${i ? "L" : "M"}${x(ratios[i])} ${y(v)}`).join("");
  const yTicks = [y0 + pad * 0.5, (y0 + y1) / 2, y1 - pad * 0.5];

  return (
    <svg
      viewBox="0 0 1000 420"
      role="img"
      aria-label={`${policy}'s savings across cost ratios ${ratios[0]} to ${ratios[ratios.length - 1]}, staying between ${band ? fmtNum(band[0], 4) : "—"} and ${band ? fmtNum(band[1], 4) : "—"}, above the volume_rank floor at every ratio.`}
      className="block h-auto w-full"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {/* grid + axes */}
      <motion.g style={{ opacity: backdrop }}>
        {ratios.map((r) => (
          <g key={r}>
            <path d={`M${x(r)} ${PY}V${PY1}`} stroke="var(--color-border)" strokeWidth="1" />
            <text x={x(r)} y={PY1 + 22} textAnchor="middle" fontSize="12" fill="var(--color-faint)">
              {r}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={t}>
            <path d={`M${PX} ${y(t)}H${PX1}`} stroke="var(--color-border)" strokeWidth="1" strokeDasharray="2 6" />
            <text x={PX - 12} y={y(t) + 4} textAnchor="end" fontSize="11" fill="var(--color-faint)">
              {fmtNum(t, 3)}
            </text>
          </g>
        ))}
        <text x={(PX + PX1) / 2} y={PY1 + 44} textAnchor="middle" fontSize="11" fill="var(--color-faint)" letterSpacing="0.12em">
          COST OF A WRONG HOLD ÷ COST OF A MISSED FRAUD · LOG SCALE
        </text>
        <text x={PX - 12} y={PY - 14} textAnchor="end" fontSize="11" fill="var(--color-faint)" letterSpacing="0.12em">
          SAVINGS
        </text>
      </motion.g>

      {/* the other policies, faint */}
      {otherSeries.map((s) => (
        <motion.path
          key={s.policy}
          d={line(s.values)}
          fill="none"
          stroke="var(--color-muted-foreground)"
          strokeWidth="1.5"
          style={{ pathLength: others, opacity: 0.3 }}
        />
      ))}

      {/* the floor */}
      {floorSeries && (
        <>
          <motion.path
            d={line(floorSeries.values)}
            fill="none"
            stroke="var(--color-foreground)"
            strokeWidth="2"
            strokeDasharray="6 6"
            style={{ pathLength: floor, opacity: 0.75 }}
          />
          <motion.text
            x={PX1 + 8}
            y={y(floorSeries.values[floorSeries.values.length - 1]) + 4}
            fontSize="11"
            fontWeight="700"
            fill="var(--color-foreground)"
            style={{ opacity: floor }}
          >
            {floorSeries.policy}
          </motion.text>
        </>
      )}

      {/* the shipped operating point */}
      {shippedRatio != null && shippedWithin && (
        <motion.g style={{ opacity: marker }}>
          <path d={`M${x(shippedRatio)} ${PY}V${PY1}`} stroke="var(--color-notice)" strokeWidth="1.5" strokeDasharray="3 5" />
          <text x={x(shippedRatio) + 8} y={PY + 14} fontSize="11" fill="var(--color-notice)" letterSpacing="0.1em">
            SHIPPED · {fmtNum(shippedRatio, 3)}
          </text>
        </motion.g>
      )}

      {/* the survivor */}
      <motion.path
        d={line(series.values)}
        fill="none"
        stroke="var(--color-primary)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ pathLength: hero, filter: "drop-shadow(0 0 10px var(--color-primary))" }}
      />
      <motion.g style={{ opacity: points }}>
        {series.values.map((v, i) => (
          <g key={ratios[i]}>
            <circle cx={x(ratios[i])} cy={y(v)} r={6} fill="var(--color-background)" stroke="var(--color-primary)" strokeWidth="3" />
            <text x={x(ratios[i])} y={y(v) + 26} textAnchor="middle" fontSize="12" fontWeight="700" fill="var(--color-primary-text)">
              {fmtNum(v, 4)}
            </text>
          </g>
        ))}
        <text x={PX1 + 8} y={y(series.values[series.values.length - 1]) + 4} fontSize="11" fontWeight="700" fill="var(--color-primary-text)">
          {policy}
        </text>
      </motion.g>
    </svg>
  );
}

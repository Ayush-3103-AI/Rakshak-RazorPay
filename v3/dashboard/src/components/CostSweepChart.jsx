// The cost-asymmetry sweep (#79): every policy's savings across four orders of
// magnitude of false-hold/fraud-loss asymmetry.
//
// The x-axis is LOG-scaled because the swept grid is [0.01 … 100]; on a linear
// axis four of the five points collapse against the origin and the flatness that
// is the entire finding becomes invisible. The shipped cost matrix is drawn as a
// reference line so a reader can see where the operating point sits rather than
// take it from prose — `shipped_ratio_within_grid` on the artifact is what says
// whether that line is inside the swept range at all.
//
// Series are distinguished by dash pattern as well as hue, so the figure survives
// grayscale, a projector, and colorblind viewing.
import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtNum } from "../lib/format.js";

// Five slots, cycled. Colors are the panel's own semantic tokens rather than a
// second palette, so the chart stays inside the Blade-derived system.
const STROKES = [
  { color: "var(--color-primary)", dash: undefined },
  { color: "var(--color-information)", dash: "7 4" },
  { color: "var(--color-notice)", dash: "2 3" },
  { color: "var(--color-positive)", dash: "10 4 2 4" },
  { color: "var(--color-negative)", dash: "4 4" },
];

function SweepTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[var(--radius-sm)] border border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-3)] text-xs shadow-[var(--shadow-mid)]">
      <p className="m-0 mb-[var(--spacing-2)] font-mono font-semibold text-foreground">
        asymmetry ratio {label}
      </p>
      {[...payload]
        .sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity))
        .map((p) => (
          <p key={p.dataKey} className="m-0 flex items-center gap-[var(--spacing-2)] text-muted-foreground">
            <span className="inline-block h-[8px] w-[8px] rounded-full" style={{ background: p.stroke }} />
            <span className="font-mono">{p.name}</span>: {fmtNum(p.value, 4)}
          </p>
        ))}
    </div>
  );
}

export default function CostSweepChart({ ratios, series, shippedRatio, shippedWithinGrid, animate = true }) {
  const data = useMemo(
    () =>
      ratios.map((ratio, i) => {
        const row = { ratio };
        for (const s of series) row[s.policy] = s.values[i];
        return row;
      }),
    [ratios, series]
  );

  return (
    // Fills its container rather than pinning 340px: under the snap-paged shell
    // this figure is the whole point of its screen, and a fixed height either
    // strands it in dead space on a tall display or overflows a short one. The
    // floor keeps it readable when the viewport really is small.
    <div className="chart-inner h-full min-h-[260px] min-w-[560px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 28, bottom: 12, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="ratio"
            type="number"
            scale="log"
            domain={["dataMin", "dataMax"]}
            ticks={ratios}
            tickFormatter={(v) => String(v)}
            tick={{ fontSize: 11, fill: "var(--color-faint)" }}
            stroke="var(--color-border-strong)"
            label={{
              value: "false-hold cost ÷ mean fraud loss  (log scale)",
              position: "insideBottom",
              offset: -6,
              fontSize: 11,
              fill: "var(--color-faint)",
            }}
          />
          <YAxis
            tickFormatter={(v) => fmtNum(v, 2)}
            tick={{ fontSize: 11, fill: "var(--color-faint)" }}
            stroke="var(--color-border-strong)"
            width={56}
            label={{
              value: "savings",
              angle: -90,
              position: "insideLeft",
              fontSize: 11,
              fill: "var(--color-faint)",
            }}
          />

          {shippedWithinGrid && shippedRatio != null && (
            <ReferenceLine
              x={shippedRatio}
              stroke="var(--color-foreground)"
              strokeDasharray="5 4"
              strokeOpacity={0.55}
              label={{
                value: "shipped cost matrix",
                position: "insideTopLeft",
                fontSize: 10,
                fill: "var(--color-faint)",
              }}
            />
          )}

          {series.map((s, i) => {
            const { color, dash } = STROKES[i % STROKES.length];
            return (
              <Line
                key={s.policy}
                type="monotone"
                dataKey={s.policy}
                name={s.policy}
                stroke={color}
                strokeWidth={2.25}
                strokeDasharray={dash}
                dot={{ r: 2.5, strokeWidth: 0, fill: color }}
                isAnimationActive={animate}
                animationDuration={900}
                animationBegin={i * 90}
                animationEasing="ease-out"
              />
            );
          })}

          <Legend
            wrapperStyle={{ fontSize: 11, fontFamily: "var(--font-mono)", paddingTop: 8 }}
            iconType="plainline"
          />
          <Tooltip content={<SweepTooltip />} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

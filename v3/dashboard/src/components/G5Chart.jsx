// The G5 confounder-null figure (#62): raw vs cohort-residual alert rate at
// prevalence 0, with the confounder windows shaded by their own `role` field
// — never a hardcoded window list. Windows are half-open [start_day, end_day)
// per the artefact's own `window_convention`; ReferenceArea's x2 lands
// exactly on that boundary on this continuous day axis, so a one-day window
// shades one day wide, not two.
//
// Adversarial and control windows are distinguished by fill PATTERN as well
// as color (a diagonal hatch vs a dot grid) so the encoding survives
// grayscale and colorblind viewing, not only hue.
import { useId, useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtPct } from "../lib/format.js";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[var(--radius-sm)] border border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-3)] text-xs shadow-[var(--shadow-mid)]">
      <p className="m-0 mb-[var(--spacing-2)] font-mono font-semibold text-foreground">day {label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} className="m-0 flex items-center gap-[var(--spacing-2)] text-muted-foreground">
          <span className="inline-block h-[8px] w-[8px] rounded-full" style={{ background: p.stroke }} />
          {p.name}: {p.value == null ? "no baseline yet" : fmtPct(p.value, 2)}
        </p>
      ))}
    </div>
  );
}

export default function G5Chart({ nDays, windows, series, nominalAlertRate, excessAllowedPp }) {
  const hatchId = useId();
  const dotsId = useId();

  const data = useMemo(() => {
    const rows = [];
    for (let day = 0; day < nDays; day += 1) {
      const row = { day };
      for (const s of series) row[s.detector] = s.alert_rate_by_day[day] ?? null;
      rows.push(row);
    }
    return rows;
  }, [nDays, series]);

  const ticks = useMemo(() => {
    const step = Math.max(1, Math.round(nDays / 12));
    const out = [];
    for (let d = 0; d < nDays; d += step) out.push(d);
    return out;
  }, [nDays]);

  const ceiling = nominalAlertRate != null && excessAllowedPp != null ? nominalAlertRate + excessAllowedPp / 100 : null;

  const raw = series.find((s) => s.detector === "raw");
  const residual = series.find((s) => s.detector === "cohort-residual");

  return (
    // Fills its container — see the note in CostSweepChart.jsx.
    <div className="chart-inner h-full min-h-[260px] min-w-[560px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 12, right: 24, bottom: 8, left: 4 }}>
          <defs>
            <pattern id={hatchId} width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
              <rect width="7" height="7" fill="var(--color-negative-bg)" />
              <line x1="0" y1="0" x2="0" y2="7" stroke="var(--color-negative)" strokeWidth="1.6" strokeOpacity="0.55" />
            </pattern>
            <pattern id={dotsId} width="9" height="9" patternUnits="userSpaceOnUse">
              <rect width="9" height="9" fill="var(--color-information-bg)" />
              <circle cx="2" cy="2" r="1.2" fill="var(--color-information)" fillOpacity="0.6" />
            </pattern>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
          <XAxis
            dataKey="day"
            type="number"
            domain={[0, nDays - 1]}
            ticks={ticks}
            tick={{ fontSize: 11, fill: "var(--color-faint)" }}
            stroke="var(--color-border-strong)"
            label={{ value: "simulation day", position: "insideBottom", offset: -4, fontSize: 11, fill: "var(--color-faint)" }}
          />
          <YAxis
            tickFormatter={(v) => fmtPct(v, 1)}
            tick={{ fontSize: 11, fill: "var(--color-faint)" }}
            stroke="var(--color-border-strong)"
            width={56}
            label={{ value: "alert rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--color-faint)" }}
          />

          {windows.map((w, i) => (
            <ReferenceArea
              key={`${w.confounder}-${w.start_day}-${i}`}
              x1={w.start_day}
              x2={w.end_day}
              fill={`url(#${w.role === "adversarial" ? hatchId : dotsId})`}
              stroke={w.role === "adversarial" ? "var(--color-negative)" : "var(--color-information)"}
              strokeOpacity={0.4}
              label={{
                value: w.confounder,
                position: "insideTop",
                fontSize: 10,
                fill: w.role === "adversarial" ? "var(--color-negative)" : "var(--color-information)",
              }}
            />
          ))}

          {ceiling != null && (
            <ReferenceLine
              y={ceiling}
              stroke="var(--color-notice)"
              strokeDasharray="5 4"
              label={{
                value: `excess ceiling (+${excessAllowedPp}pp)`,
                position: "insideTopRight",
                fontSize: 10,
                fill: "var(--color-notice)",
              }}
            />
          )}
          {nominalAlertRate != null && (
            <ReferenceLine
              y={nominalAlertRate}
              stroke="var(--color-faint)"
              strokeDasharray="2 3"
              label={{ value: "nominal", position: "insideBottomRight", fontSize: 10, fill: "var(--color-faint)" }}
            />
          )}

          {raw && (
            <Line
              type="monotone"
              dataKey="raw"
              name="raw"
              stroke="var(--color-negative)"
              strokeWidth={2.25}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          )}
          {residual && (
            <Line
              type="monotone"
              dataKey="cohort-residual"
              name="cohort-residual"
              stroke="var(--color-information)"
              strokeWidth={2.25}
              strokeDasharray="7 4"
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          )}
          <Tooltip content={<ChartTooltip />} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

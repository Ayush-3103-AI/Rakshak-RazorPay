// The mechanism, as a flow: merchant-days come in on the left, are scored,
// pass a decision gate that is capped at K a day, and leave in one of three
// lanes. Only the two non-PASS lanes carry the "reason attached" mark, which
// is the product promise this screen is making.
//
// Driven by a 0→1 MotionValue, all strokes via `pathLength`. The lane lengths
// are an illustration of proportion, not counts — the artifacts do not record
// per-action volumes, so none are drawn. `k` and `features` are the only
// figures here, and both arrive from the artifacts.
import { motion, useTransform } from "framer-motion";

const IN_X = 60;
const SCORE_X = 330;
const GATE_X = 560;
const LANE_X = 760;

function grid(cols, rows, x0, y0, step) {
  let d = "";
  for (let r = 0; r < rows; r += 1) for (let c = 0; c < cols; c += 1) d += `M${x0 + c * step} ${y0 + r * step}h0.01`;
  return d;
}

function Box({ x, y, w, h, title, sub, tone = "var(--color-primary-text)", opacity }) {
  return (
    <motion.g style={{ opacity }}>
      <rect x={x} y={y} width={w} height={h} rx={16} fill="var(--glass-strong)" stroke="var(--glass-border)" />
      <rect x={x} y={y} width={w} height={1} fill="var(--glass-highlight)" />
      <text x={x + w / 2} y={y + h / 2 - 6} textAnchor="middle" fontSize="15" fontWeight="700" fill="var(--color-foreground)" fontFamily="var(--font-heading)">
        {title}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 16} textAnchor="middle" fontSize="10" fill={tone} letterSpacing="0.06em">
        {sub}
      </text>
    </motion.g>
  );
}

function Lane({ y, length, label, tone, reason, draw, labelIn }) {
  return (
    <>
      <motion.path
        d={`M${LANE_X} ${y}h${length}`}
        stroke={tone}
        strokeWidth="14"
        strokeLinecap="round"
        style={{ pathLength: draw, opacity: 0.9 }}
      />
      <motion.g style={{ opacity: labelIn }}>
        <text x={LANE_X} y={y - 16} textAnchor="start" fontSize="13" fontWeight="700" fill={tone} letterSpacing="0.12em">
          {label}
        </text>
        {reason && (
          <g>
            <rect x={LANE_X + length + 16} y={y - 11} width={112} height={22} rx={11} fill="var(--color-notice-bg)" stroke="var(--color-notice-border)" />
            <text x={LANE_X + length + 72} y={y + 4} textAnchor="middle" fontSize="10" fontWeight="700" fill="var(--color-notice)" letterSpacing="0.1em">
              + REASON
            </text>
          </g>
        )}
      </motion.g>
    </>
  );
}

export default function Pipeline({ progress, k, features }) {
  const inflow = useTransform(progress, [0, 0.22], [0, 1]);
  const toScore = useTransform(progress, [0.18, 0.34], [0, 1]);
  const score = useTransform(progress, [0.26, 0.4], [0, 1]);
  const toGate = useTransform(progress, [0.4, 0.54], [0, 1]);
  const gate = useTransform(progress, [0.48, 0.62], [0, 1]);
  const fan = useTransform(progress, [0.6, 0.74], [0, 1]);
  const lanes = useTransform(progress, [0.7, 0.92], [0, 1]);
  const laneLabels = useTransform(progress, [0.84, 1], [0, 1]);

  const kText = Number.isFinite(k) ? `CAPPED AT K = ${k} A DAY` : "CAPPED AT K A DAY";
  const fText = Number.isFinite(features) ? `${features} FEATURES · 1 SCORE` : "FEATURES → 1 SCORE";

  return (
    <svg
      viewBox="0 0 1000 380"
      role="img"
      aria-label={`Merchant-days flow into a scorer, then a decision gate ${Number.isFinite(k) ? `capped at ${k} alerts a day` : "with a daily cap"}, and leave as PASS, REVIEW or HOLD. REVIEW and HOLD carry a reason.`}
      className="block h-auto w-full"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {/* incoming merchant-days */}
      <motion.path
        d={grid(6, 5, IN_X, 110, 30)}
        stroke="var(--color-faint)"
        strokeWidth="9"
        strokeLinecap="round"
        style={{ pathLength: inflow }}
      />
      <motion.text x={IN_X} y={276} fontSize="11" fill="var(--color-faint)" letterSpacing="0.12em" style={{ opacity: inflow }}>
        EVERY CLEARED MERCHANT
      </motion.text>
      <motion.text x={IN_X} y={294} fontSize="11" fill="var(--color-faint)" letterSpacing="0.12em" style={{ opacity: inflow }}>
        EVERY DAY
      </motion.text>

      {/* → score */}
      <motion.path d={`M${IN_X + 180} 170H${SCORE_X - 12}`} stroke="var(--color-border-strong)" strokeWidth="2" strokeLinecap="round" style={{ pathLength: toScore }} />
      <Box x={SCORE_X} y={130} w={170} h={80} title="Score" sub={fText} opacity={score} />

      {/* → gate */}
      <motion.path d={`M${SCORE_X + 182} 170H${GATE_X - 12}`} stroke="var(--color-border-strong)" strokeWidth="2" strokeLinecap="round" style={{ pathLength: toGate }} />
      <Box x={GATE_X} y={130} w={170} h={80} title="Decide" sub={kText} tone="var(--color-notice)" opacity={gate} />

      {/* fan-out */}
      <motion.path
        d={`M${GATE_X + 182} 170C${GATE_X + 214} 170 ${GATE_X + 214} 100 ${LANE_X - 14} 100M${GATE_X + 182} 170H${LANE_X - 14}M${GATE_X + 182} 170C${GATE_X + 214} 170 ${GATE_X + 214} 240 ${LANE_X - 14} 240`}
        fill="none"
        stroke="var(--color-border-strong)"
        strokeWidth="2"
        strokeLinecap="round"
        style={{ pathLength: fan }}
      />

      <Lane y={100} length={150} label="PASS" tone="var(--color-faint)" draw={lanes} labelIn={laneLabels} />
      <Lane y={170} length={70} label="REVIEW" tone="var(--color-notice)" reason draw={lanes} labelIn={laneLabels} />
      <Lane y={240} length={26} label="HOLD" tone="var(--color-negative)" reason draw={lanes} labelIn={laneLabels} />

      <motion.text x={IN_X} y={352} fontSize="11" fill="var(--color-faint)" letterSpacing="0.12em" style={{ opacity: laneLabels }}>
        LANE LENGTHS ILLUSTRATE PROPORTION, NOT COUNTS
      </motion.text>
    </svg>
  );
}

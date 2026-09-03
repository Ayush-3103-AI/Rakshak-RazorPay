// The merchant-lifetime axis: one line, and the emptiness on it is the argument.
//
// Bumblebee is a single stamp at day zero. Vulcan is the dense comb of ticks
// under the line — every transaction, scored in milliseconds, for the whole
// life of the merchant. Nothing sits ABOVE the line between the stamp and the
// day a chargeback lands, and that is the hole Rakshak fills: the row of daily
// dots that appears last.
//
// Driven by a 0→1 MotionValue. Everything that draws is a path with
// `pathLength`, so a comb of a hundred ticks is one element and appears left
// to right as one stroke; at progress 1 the figure is complete and static,
// which is what the no-motion and narrow-viewport paths render.
//
// The day positions are an illustration of the product timeline, not a
// measurement — the only figures on this screen are words.
import { motion, useTransform } from "framer-motion";

const X0 = 70; // day 0
const X1 = 940; // day 130
const PX_PER_DAY = (X1 - X0) / 130;
const day = (d) => X0 + d * PX_PER_DAY;
const AXIS_Y = 196;

const CHARGEBACK_DAY = 92;
const WINDOW = [45, 120];

function comb(from, to, step, y1, y2) {
  let d = "";
  for (let x = from; x <= to; x += step) d += `M${x} ${y1}V${y2}`;
  return d;
}

function dots(from, to, step, y) {
  let d = "";
  for (let x = from; x <= to; x += step) d += `M${x} ${y}h0.01`;
  return d;
}

export default function GapAxis({ progress }) {
  const axis = useTransform(progress, [0, 0.16], [0, 1]);
  const labels = useTransform(progress, [0.08, 0.2], [0, 1]);
  const stamp = useTransform(progress, [0.12, 0.24], [0, 1]);
  const vulcan = useTransform(progress, [0.2, 0.46], [0, 1]);
  const bracket = useTransform(progress, [0.46, 0.6], [0, 1]);
  const gapLabel = useTransform(progress, [0.54, 0.66], [0, 1]);
  const band = useTransform(progress, [0.62, 0.74], [0, 1]);
  const flag = useTransform(progress, [0.7, 0.8], [0, 1]);
  const rakshak = useTransform(progress, [0.8, 0.98], [0, 1]);
  const rakshakLabel = useTransform(progress, [0.88, 1], [0, 1]);

  return (
    <svg
      viewBox="0 0 1000 330"
      role="img"
      aria-label="A merchant's lifetime on one axis: Bumblebee reviews once at day zero, Vulcan scores every transaction along the way, and nothing watches the merchant until a chargeback lands 45 to 120 days later. Rakshak adds one decision per day across that gap."
      className="block h-auto w-full"
      style={{ fontFamily: "var(--font-mono)" }}
    >
      {/* chargeback window */}
      <motion.rect
        x={day(WINDOW[0])}
        y={AXIS_Y - 26}
        width={day(WINDOW[1]) - day(WINDOW[0])}
        height={52}
        rx={8}
        fill="var(--color-negative)"
        style={{ opacity: useTransform(band, [0, 1], [0, 0.12]) }}
      />
      <motion.text
        x={day(WINDOW[1])}
        y={AXIS_Y + 70}
        textAnchor="end"
        fontSize="11"
        fill="var(--color-negative)"
        letterSpacing="0.14em"
        style={{ opacity: band }}
      >
        CHARGEBACKS LAND · DAYS 45–120
      </motion.text>

      {/* the axis */}
      <motion.path
        d={`M${X0} ${AXIS_Y}H${X1}`}
        stroke="var(--color-border-strong)"
        strokeWidth="2"
        strokeLinecap="round"
        style={{ pathLength: axis }}
      />
      <motion.g style={{ opacity: labels }} fill="var(--color-faint)" fontSize="12">
        <text x={X0} y={AXIS_Y + 96} textAnchor="start">
          day 0
        </text>
        <text x={day(45)} y={AXIS_Y + 96} textAnchor="middle">
          day 45
        </text>
        <text x={day(120)} y={AXIS_Y + 96} textAnchor="middle">
          day 120
        </text>
      </motion.g>

      {/* Vulcan: the comb under the line */}
      <motion.path
        d={comb(X0, X1, 8, AXIS_Y + 8, AXIS_Y + 26)}
        stroke="var(--color-information)"
        strokeWidth="2"
        strokeLinecap="round"
        style={{ pathLength: vulcan, opacity: 0.85 }}
      />
      <motion.text
        x={X1}
        y={AXIS_Y + 50}
        textAnchor="end"
        fontSize="11"
        fill="var(--color-information)"
        letterSpacing="0.14em"
        style={{ opacity: useTransform(vulcan, [0.3, 1], [0, 1]) }}
      >
        VULCAN · SCORES EVERY TRANSACTION, IN MILLISECONDS
      </motion.text>

      {/* Bumblebee: one stamp at day zero */}
      <motion.g style={{ opacity: stamp }}>
        <circle cx={X0} cy={AXIS_Y} r={15} fill="var(--color-background)" stroke="var(--color-positive)" strokeWidth="2.5" />
        <path
          d={`M${X0 - 6} ${AXIS_Y}l4 4 8-9`}
          fill="none"
          stroke="var(--color-positive)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <text x={X0} y={AXIS_Y + 50} textAnchor="start" fontSize="11" fill="var(--color-positive)" letterSpacing="0.14em">
          BUMBLEBEE · REVIEWS ONCE, AT ONBOARDING
        </text>
      </motion.g>

      {/* the gap: a bracket over everything between the stamp and the chargeback */}
      <motion.path
        d={`M${X0 + 24} 78V66H${day(CHARGEBACK_DAY) - 18}V78`}
        fill="none"
        stroke="var(--color-notice)"
        strokeWidth="1.5"
        strokeLinecap="round"
        style={{ pathLength: bracket }}
      />
      <motion.text
        x={(X0 + day(CHARGEBACK_DAY)) / 2}
        y={52}
        textAnchor="middle"
        fontSize="13"
        fontWeight="700"
        fill="var(--color-notice)"
        letterSpacing="0.16em"
        style={{ opacity: gapLabel }}
      >
        NOTHING WATCHES THE MERCHANT
      </motion.text>

      {/* the chargeback */}
      <motion.g style={{ opacity: flag }}>
        <path
          d={`M${day(CHARGEBACK_DAY)} ${AXIS_Y}V98`}
          stroke="var(--color-negative)"
          strokeWidth="2"
          strokeLinecap="round"
        />
        <path
          d={`M${day(CHARGEBACK_DAY)} 98h26l-8 9 8 9h-26z`}
          fill="var(--color-negative)"
        />
        <text
          x={day(CHARGEBACK_DAY) + 34}
          y={112}
          fontSize="11"
          fill="var(--color-negative)"
          letterSpacing="0.14em"
        >
          FIRST SIGNAL
        </text>
      </motion.g>

      {/* Rakshak: a decision every day, across the gap and beyond */}
      <motion.path
        d={dots(X0 + 20, X1 - 6, 12, AXIS_Y - 40)}
        stroke="var(--color-primary)"
        strokeWidth="7"
        strokeLinecap="round"
        style={{ pathLength: rakshak }}
      />
      <motion.text
        x={X1}
        y={AXIS_Y - 56}
        textAnchor="end"
        fontSize="12"
        fontWeight="700"
        fill="var(--color-primary-text)"
        letterSpacing="0.14em"
        style={{ opacity: rakshakLabel }}
      >
        RAKSHAK · ONE DECISION PER MERCHANT, PER DAY
      </motion.text>
    </svg>
  );
}

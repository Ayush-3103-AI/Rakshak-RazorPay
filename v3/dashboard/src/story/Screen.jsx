// One screen of the story, and the handful of pieces every screen is made of.
//
// A screen is a full-viewport section whose children reveal in a stagger the
// first time it scrolls into view: each piece rises, un-blurs and settles, in
// reading order, so the type scale and the motion agree about what to read
// first. The reveal runs once — scrolling back up should not replay a page
// that has already been read.
//
// A screen may PIN. Pass `pin={240}` and the section becomes 240vh tall with
// its content stuck to the viewport, and `children` is called with a
// MotionValue that runs 0→1 across that height. That is CSS `position: sticky`
// plus a scroll read — no wheel handler, no preventDefault, nothing owning
// the scrollbar. Find-in-page, Page Down and screen readers all keep working,
// which is the line this project drew against v1's playhead and will not
// cross again. Under reduced motion, or below 900px, the pin is dropped and
// the figure renders in its settled state.
import { motion, useMotionValue, useReducedMotion, useScroll } from "framer-motion";
import { useRef } from "react";
import Counter from "../components/Counter.jsx";
import { cn } from "../lib/cn.js";
import { useCinematic } from "./useCinematic.js";

export const EASE = [0.2, 0.65, 0.2, 1];

export const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.1, delayChildren: 0.08 } },
};

export const item = {
  hidden: { opacity: 0, y: 32, filter: "blur(10px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.85, ease: EASE } },
};

/** Anything that should take part in its screen's stagger. */
export function Reveal({ as = "div", className, children, ...rest }) {
  const Tag = motion[as];
  return (
    <Tag variants={item} className={className} {...rest}>
      {children}
    </Tag>
  );
}

export function Eyebrow({ children, className }) {
  return (
    <Reveal
      as="p"
      className={cn(
        "m-0 font-mono text-[11px] font-bold tracking-[0.24em] text-primary-text uppercase",
        className
      )}
    >
      {children}
    </Reveal>
  );
}

const HEADLINE_SIZE = {
  xl: "text-[length:var(--text-display)] leading-[0.94]",
  lg: "text-[length:var(--text-display-md)] leading-[0.98]",
  md: "text-[length:var(--text-display-sm)] leading-[1.02]",
};

export function Headline({ as = "h2", size = "lg", children, className }) {
  return (
    <Reveal
      as={as}
      className={cn(
        "m-0 mt-[var(--spacing-5)] max-w-[15ch] font-heading font-extrabold tracking-[-0.035em] text-balance text-foreground",
        HEADLINE_SIZE[size],
        className
      )}
    >
      {children}
    </Reveal>
  );
}

export function Lede({ children, className }) {
  return (
    <Reveal
      as="p"
      className={cn(
        "m-0 mt-[var(--spacing-7)] max-w-[56ch] text-[length:var(--text-lede)] leading-[1.55] text-muted-foreground",
        className
      )}
    >
      {children}
    </Reveal>
  );
}

/** A frosted panel. `flat` for panels nested inside another glass panel. */
export function Glass({ as = "div", flat = false, className, children, ...rest }) {
  return (
    <Reveal
      as={as}
      className={cn(
        flat ? "glass-flat" : "glass",
        "rounded-[var(--radius-3xl)] p-[clamp(20px,2.4vw,36px)]",
        className
      )}
      {...rest}
    >
      {children}
    </Reveal>
  );
}

const CHIP_TONE = {
  muted: "border-border bg-canvas-well/60 text-muted-foreground",
  primary: "border-primary/40 bg-primary/15 text-primary-text",
  notice: "border-notice-border bg-notice-bg text-notice",
  negative: "border-negative-border bg-negative-bg text-negative",
  positive: "border-positive-border bg-positive-bg text-positive",
};

/** A small mono label. Not a Reveal: chips sit inside one. */
export function Chip({ tone = "muted", className, children }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-[var(--spacing-2)] rounded-full border px-[var(--spacing-5)] py-[var(--spacing-3)] font-mono text-[11px] font-bold tracking-[0.14em] uppercase",
        CHIP_TONE[tone],
        className
      )}
    >
      {children}
    </span>
  );
}

/** A big number with a label under it. `value` may be a number (counts up) or a string. */
export function Stat({ value, format, label, note, size = "md", className, accent = false }) {
  const numeric = typeof value === "number";
  const cls = cn(
    "block font-mono leading-none font-bold tracking-[-0.03em] tabular-nums",
    size === "lg"
      ? "text-[length:var(--text-stat)]"
      : size === "xs"
        ? "text-[clamp(26px,2.4vw,38px)]"
        : "text-[length:var(--text-stat-sm)]",
    accent ? "text-primary-text" : "text-foreground"
  );
  return (
    <div className={cn("min-w-0", className)}>
      {numeric ? (
        <Counter value={value} format={format ?? ((v) => Math.round(v).toLocaleString("en-IN"))} className={cls} />
      ) : (
        <span className={cls}>{value}</span>
      )}
      <p className="m-0 mt-[var(--spacing-4)] text-base leading-snug font-semibold text-foreground">{label}</p>
      {note && <p className="m-0 mt-[var(--spacing-2)] text-sm leading-snug text-faint">{note}</p>}
    </div>
  );
}

export default function Screen({ id, pin, children, className, contentClassName }) {
  const reduce = useReducedMotion();
  const ref = useRef(null);
  const cinematic = useCinematic() && Boolean(pin);
  // Always attached to the section and always subscribed: a target ref that
  // is sometimes unmounted is a warning from framer-motion, and the read costs
  // nothing when the value is unused.
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const settled = useMotionValue(1);
  const progress = cinematic ? scrollYProgress : settled;
  const body = typeof children === "function" ? children(progress) : children;

  return (
    <section
      ref={ref}
      id={id}
      data-screen={id}
      style={cinematic ? { height: `${pin}vh` } : undefined}
      className={cn("relative w-full", className)}
    >
      <div
        className={cn(
          "flex w-full items-center px-[clamp(20px,5vw,72px)] py-[clamp(72px,10vh,112px)]",
          cinematic ? "sticky top-0 h-screen overflow-hidden" : "min-h-screen"
        )}
      >
        <motion.div
          className={cn("mx-auto w-full max-w-[1320px]", contentClassName)}
          variants={container}
          initial={reduce ? false : "hidden"}
          whileInView="show"
          viewport={{ once: true, amount: 0.2 }}
        >
          {body}
        </motion.div>
      </div>
    </section>
  );
}

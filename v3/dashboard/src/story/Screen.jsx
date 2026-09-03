// One screen of the story, and the handful of pieces every screen is made of.
//
// A screen is exactly one viewport tall and it is a snap stop: one gesture
// takes you to the next screen and only the next screen (tokens.css). Content
// that will not fit scrolls INSIDE the screen, and the outer snap resumes once
// that inner scroller is exhausted — so nothing is ever unreachable, and the
// page still advances a screen at a time.
//
// Children reveal in a stagger the first time the screen arrives: each piece
// rises, un-blurs and settles, in reading order, so the type scale and the
// motion agree about what to read first. The reveal runs once — coming back to
// a screen you have already read should not replay it.
//
// THE INNER SCROLLER IS A FALLBACK, NOT A LAYOUT. Every screen is authored to
// fit one viewport at a normal desktop height, and it is worth the effort:
// even 40px of inner overflow means a wheel gesture scrolls a little way
// INSIDE the screen before the page advances, which reads to a person as the
// page lock failing. The scroller exists so that a short window or a long
// translation is still readable, not as licence to overfill a screen.
//
// THE FIGURES ARE PLAYED, NOT SCRUBBED. Pass `play={2.4}` and `children` is
// called with a MotionValue that runs 0→1 over 2.4 seconds when the screen
// arrives. This replaces the earlier scroll-scrubbed pinning, which required a
// section several viewports tall and is therefore incompatible with locking
// one gesture to one page. The figures draw exactly the same sequence; what
// changed is the clock driving them. Under reduced motion the value starts at
// 1 and the figure renders complete, which is also what it does if the tween
// is interrupted — the settled state is always the truth.
import { animate, motion, useInView, useMotionValue, useReducedMotion } from "framer-motion";
import { useEffect, useRef } from "react";
import Counter from "../components/Counter.jsx";
import { cn } from "../lib/cn.js";

export const EASE = [0.2, 0.65, 0.2, 1];

export const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.09, delayChildren: 0.06 } },
};

export const item = {
  hidden: { opacity: 0, y: 28, filter: "blur(10px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.8, ease: EASE } },
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
        "m-0 mt-[var(--spacing-4)] max-w-[15ch] font-heading font-extrabold tracking-[-0.035em] text-balance text-foreground",
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
        "m-0 mt-[var(--spacing-6)] max-w-[56ch] text-[length:var(--text-lede)] leading-[1.5] text-muted-foreground",
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
        "rounded-[var(--radius-2xl)] p-[clamp(16px,2vw,30px)]",
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
        "inline-flex items-center gap-[var(--spacing-2)] rounded-full border px-[var(--spacing-4)] py-[var(--spacing-2)] font-mono text-[11px] font-bold tracking-[0.14em] uppercase",
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
        ? "text-[clamp(24px,2.2vw,34px)]"
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
      <p className="m-0 mt-[var(--spacing-3)] text-sm leading-snug font-semibold text-foreground">{label}</p>
      {note && <p className="m-0 mt-[var(--spacing-1)] text-xs leading-snug text-faint">{note}</p>}
    </div>
  );
}

export default function Screen({ id, play, children, className, contentClassName }) {
  const reduce = useReducedMotion();
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.35 });
  const progress = useMotionValue(play && !reduce ? 0 : 1);

  useEffect(() => {
    if (!play) return undefined;
    if (reduce) {
      progress.set(1);
      return undefined;
    }
    if (!inView) return undefined;
    const controls = animate(progress, 1, { duration: play, ease: "linear" });
    return () => controls.stop();
  }, [inView, play, reduce, progress]);

  const body = typeof children === "function" ? children(progress) : children;

  return (
    <section
      ref={ref}
      id={id}
      data-screen={id}
      className={cn("relative h-screen w-full snap-start snap-always max-lg:h-auto max-lg:min-h-screen", className)}
    >
      {/* The scroller is this element, so a tall screen scrolls inside itself
          rather than pushing the snap point away from the top of the section. */}
      <div className="h-full w-full overflow-y-auto px-[clamp(20px,3.4vw,56px)] py-[clamp(40px,5vh,72px)] max-lg:pt-[92px]">
        <motion.div
          className={cn("mx-auto flex min-h-full w-full max-w-[1240px] flex-col justify-center", contentClassName)}
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

// One screen of the panel, as a card.
//
// Every page on this shell is the same three things — a mono eyebrow, one big
// declarative title, and at most one sentence under it — followed by whatever
// the page actually measures. That is not decoration: the type scale IS the
// reading order. A judge skimming at speed reads the 48px line and nothing else
// on eight screens, and the eight titles alone have to carry the argument. The
// sentence beneath is for the one reader in ten who stops.
//
// The card lifts and settles on entry rather than cross-fading. `once: false`
// is deliberate — scrolling back up should replay it, because under snap paging
// the animation is what tells you a page changed at all.
import { motion, useReducedMotion } from "framer-motion";

// `headingLevel` exists because the panel is one document, not eight: only the
// first screen carries the <h1>. Eight h1s would read to a screen reader as
// eight unrelated pages that happen to share a scroll container.
export default function Page({
  id,
  eyebrow,
  title,
  lede,
  actions,
  children,
  headingLevel: Heading = "h2",
  className = "",
}) {
  const reduce = useReducedMotion();

  return (
    <section
      id={id}
      className="flex h-screen snap-start snap-always flex-col px-[var(--spacing-7)] py-[var(--spacing-7)] max-lg:h-auto max-lg:min-h-screen max-lg:px-[var(--spacing-4)] max-lg:pt-[88px]"
    >
      <motion.div
        initial={reduce ? false : { opacity: 0, y: 26, scale: 0.985 }}
        whileInView={{ opacity: 1, y: 0, scale: 1 }}
        viewport={{ amount: 0.35, margin: "-8% 0px -8% 0px" }}
        transition={
          reduce ? { duration: 0 } : { duration: 0.5, ease: [0, 0, 0.2, 1] }
        }
        className={`glass flex min-h-0 flex-1 flex-col overflow-hidden rounded-[var(--radius-2xl)] px-[var(--spacing-9)] py-[var(--spacing-8)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-6)] ${className}`}
      >
        <header className="flex shrink-0 flex-wrap items-end justify-between gap-[var(--spacing-4)]">
          <div className="min-w-0">
            <p className="m-0 font-mono text-2xs font-bold tracking-[0.2em] text-primary-text uppercase">
              {eyebrow}
            </p>
            <Heading className="m-0 mt-[var(--spacing-3)] max-w-[20ch] font-heading text-[clamp(26px,3.1vw,46px)] leading-[1.03] font-extrabold tracking-[-0.028em] text-balance text-foreground">
              {title}
            </Heading>
            {lede && (
              <p className="m-0 mt-[var(--spacing-4)] max-w-[64ch] text-sm leading-relaxed text-muted-foreground">
                {lede}
              </p>
            )}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-[var(--spacing-3)]">{actions}</div>}
        </header>

        <div className="mt-[var(--spacing-6)] min-h-0 flex-1 overflow-y-auto">{children}</div>
      </motion.div>
    </section>
  );
}

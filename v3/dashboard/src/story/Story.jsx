// The front door: eight free-scrolling screens, a two-minute read.
//
// Gap → the product hole → the mechanism → the result → the stress test →
// what was killed → the lineage → verify. The problem first, in one breath;
// the number on screen four; the honesty after it, where a reader who has
// seen the claim goes looking for the catch.
//
// Always dark. `.story` carries the glass token set (tokens.css) whatever the
// evidence panel's toggle says, and paints its own backdrop so the body's
// theme-dependent one never shows through.
//
// Nothing here jacks the scroll: the progress bar reads it, the backdrop
// glows drift with it, three screens pin with CSS sticky and read it. Under
// reduced motion all of that settles and this is a document.
import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "../lib/cn.js";
import Gap from "./screens/Gap.jsx";
import Killed from "./screens/Killed.jsx";
import Lineage from "./screens/Lineage.jsx";
import Mechanism from "./screens/Mechanism.jsx";
import ProductHole from "./screens/ProductHole.jsx";
import Result from "./screens/Result.jsx";
import StressTest from "./screens/StressTest.jsx";
import Verify from "./screens/Verify.jsx";

export const STORY_SCREENS = [
  { id: "gap", label: "The gap" },
  { id: "product", label: "The product" },
  { id: "mechanism", label: "The mechanism" },
  { id: "result", label: "The result" },
  { id: "stress", label: "The stress test" },
  { id: "killed", label: "What we killed" },
  { id: "lineage", label: "The lineage" },
  { id: "verify", label: "Verify" },
];

function Backdrop({ progress }) {
  const reduce = useReducedMotion();
  const a = useTransform(progress, [0, 1], ["0vh", "-30vh"]);
  const b = useTransform(progress, [0, 1], ["0vh", "40vh"]);
  const c = useTransform(progress, [0, 1], ["0vh", "-18vh"]);
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-[-1] overflow-hidden" style={{ background: "linear-gradient(180deg, var(--color-background), var(--color-canvas-well))" }}>
      <motion.div
        className="absolute -top-[10%] -left-[10%] h-[70vh] w-[60vw] rounded-full"
        style={{ y: reduce ? 0 : a, background: "radial-gradient(closest-side, var(--glow-a), transparent 70%)" }}
      />
      <motion.div
        className="absolute top-[5%] -right-[10%] h-[60vh] w-[50vw] rounded-full"
        style={{ y: reduce ? 0 : b, background: "radial-gradient(closest-side, var(--glow-b), transparent 70%)" }}
      />
      <motion.div
        className="absolute -bottom-[20%] left-[15%] h-[80vh] w-[70vw] rounded-full"
        style={{ y: reduce ? 0 : c, background: "radial-gradient(closest-side, var(--glow-c), transparent 70%)" }}
      />
    </div>
  );
}

function useActiveScreen() {
  const [active, setActive] = useState(STORY_SCREENS[0].id);
  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (!visible.length) return;
        const top = visible.reduce((x, y) => (x.intersectionRatio > y.intersectionRatio ? x : y));
        setActive(top.target.getAttribute("data-screen"));
      },
      { threshold: [0.2, 0.5] }
    );
    document.querySelectorAll("[data-screen]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
  return active;
}

function Dots({ active }) {
  return (
    <nav aria-label="Story screens" className="fixed top-1/2 right-[clamp(10px,2vw,28px)] z-20 -translate-y-1/2 max-lg:hidden">
      <ol className="m-0 grid list-none gap-[var(--spacing-4)] p-0">
        {STORY_SCREENS.map((s, i) => {
          const on = s.id === active;
          return (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById(s.id)?.scrollIntoView({ block: "start", behavior: "smooth" });
                }}
                aria-current={on ? "true" : undefined}
                aria-label={`${i + 1}. ${s.label}`}
                title={s.label}
                className="group flex items-center justify-end gap-[var(--spacing-3)] no-underline"
              >
                <span
                  className={cn(
                    "font-mono text-[10px] tracking-[0.14em] uppercase transition-opacity duration-[var(--duration-quick)]",
                    "opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100",
                    on ? "text-foreground" : "text-faint"
                  )}
                >
                  {s.label}
                </span>
                <span
                  className={cn(
                    "block h-[7px] rounded-full transition-all duration-[var(--duration-moderate)]",
                    on ? "w-[22px] bg-primary" : "w-[7px] bg-border-strong group-hover:bg-faint"
                  )}
                />
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default function Story() {
  const { scrollYProgress } = useScroll();
  const reduce = useReducedMotion();
  const bar = useSpring(scrollYProgress, { stiffness: 140, damping: 26, mass: 0.4 });
  const active = useActiveScreen();

  return (
    <div className="story relative min-h-screen text-foreground">
      <Backdrop progress={scrollYProgress} />

      <motion.div
        aria-hidden="true"
        className="fixed top-0 right-0 left-0 z-40 h-[3px] origin-left bg-primary"
        style={{ scaleX: reduce ? scrollYProgress : bar }}
      />

      <header className="glass fixed top-[3px] right-0 left-0 z-30 flex h-[56px] items-center justify-between !rounded-none !border-x-0 !border-t-0 px-[clamp(16px,4vw,48px)]">
        <a href="#/" className="flex items-baseline gap-[var(--spacing-3)] no-underline">
          <span className="font-heading text-base font-extrabold tracking-tight text-foreground">RAKSHAK</span>
          <span className="font-mono text-[10px] tracking-[0.16em] text-faint uppercase max-sm:hidden">post-onboarding merchant risk sentinel</span>
        </a>
        <div className="flex items-center gap-[var(--spacing-5)]">
          <span className="font-mono text-[10px] tracking-[0.16em] text-faint uppercase max-sm:hidden">
            {STORY_SCREENS.length} screens · ~2 min
          </span>
          <a
            href="#/evidence"
            className="inline-flex items-center gap-[var(--spacing-2)] rounded-full border border-primary/40 bg-primary/15 px-[var(--spacing-5)] py-[var(--spacing-3)] font-mono text-[11px] font-bold tracking-[0.14em] text-primary-text uppercase no-underline transition-colors duration-[var(--duration-quick)] hover:bg-primary/25"
          >
            Full evidence <ArrowRight aria-hidden="true" className="h-3 w-3" />
          </a>
        </div>
      </header>

      <main>
        <Gap />
        <ProductHole />
        <Mechanism />
        <Result />
        <StressTest />
        <Killed />
        <Lineage />
        <Verify />
      </main>

      <Dots active={active} />
    </div>
  );
}

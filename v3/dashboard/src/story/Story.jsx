// The front door: eight pages, a two-minute read.
//
// Gap → the product hole → the mechanism → the result → the stress test →
// what was killed → the lineage → verify. The problem first, in one breath;
// the number on page four; the honesty after it, where a reader who has seen
// the claim goes looking for the catch.
//
// ONE DELIBERATE PUSH, ONE PAGE. `usePagedScroll` catches the wheel and moves
// exactly one page; CSS snapping (tokens.css) stays underneath as the safety
// net for every other way of moving — Page Down, Home, End, find-in-page, a
// clicked page title. Nothing else is intercepted, so the document still
// scrolls and its position is still real.
import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import Brand from "../components/Brand.jsx";
import Rail, { RAIL_WIDTH, RailChips } from "../components/Rail.jsx";
import TopBar from "../components/TopBar.jsx";
import { usePagedScroll } from "../usePagedScroll.js";
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

const SCREEN_IDS = STORY_SCREENS.map((s) => s.id);

function Backdrop({ progress }) {
  const reduce = useReducedMotion();
  const a = useTransform(progress, [0, 1], ["0vh", "-30vh"]);
  const b = useTransform(progress, [0, 1], ["0vh", "40vh"]);
  const c = useTransform(progress, [0, 1], ["0vh", "-18vh"]);
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-[-1] overflow-hidden"
      style={{ background: "linear-gradient(180deg, var(--color-background), var(--color-canvas-well))" }}
    >
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
      { threshold: [0.2, 0.5, 0.75] }
    );
    document.querySelectorAll("[data-screen]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
  return active;
}

export default function Story() {
  const { scrollYProgress } = useScroll();
  const reduce = useReducedMotion();
  const bar = useSpring(scrollYProgress, { stiffness: 140, damping: 26, mass: 0.4 });
  const active = useActiveScreen();
  const ids = useMemo(() => SCREEN_IDS, []);

  usePagedScroll(ids);

  const go = useCallback((id) => {
    document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, []);

  return (
    <div className="story relative min-h-screen text-foreground">
      <Backdrop progress={scrollYProgress} />

      <motion.div
        aria-hidden="true"
        className="fixed top-0 right-0 left-0 z-40 h-[3px] origin-left bg-primary"
        style={{ scaleX: reduce ? scrollYProgress : bar }}
      />

      <TopBar
        action={
          <a
            href="#/evidence"
            className="inline-flex items-center gap-[var(--spacing-2)] rounded-full border border-primary/40 bg-primary/15 px-[var(--spacing-5)] py-[var(--spacing-3)] font-mono text-[11px] font-bold tracking-[0.14em] text-primary-text uppercase no-underline transition-colors duration-[var(--duration-quick)] hover:bg-primary/25"
          >
            Full evidence <ArrowRight aria-hidden="true" className="h-3 w-3" />
          </a>
        }
      />

      <Rail items={STORY_SCREENS} activeId={active} onSelect={go} label="Story pages" />

      <RailChips
        items={STORY_SCREENS}
        activeId={active}
        onSelect={go}
        label="Story pages"
        leading={
          <>
            <Brand href="#/" size="xs" descriptor={null} className="mr-[var(--spacing-3)] shrink-0" />
            <a
              href="#/evidence"
              className="shrink-0 rounded-full border border-primary/40 bg-primary/15 px-[var(--spacing-4)] py-[var(--spacing-2)] font-mono text-[11px] font-bold tracking-[0.1em] text-primary-text uppercase no-underline"
            >
              Evidence
            </a>
          </>
        }
      />

      <main style={{ marginRight: RAIL_WIDTH }} className="max-lg:!mr-0">
        <Gap />
        <ProductHole />
        <Mechanism />
        <Result />
        <StressTest />
        <Killed />
        <Lineage />
        <Verify />
      </main>
    </div>
  );
}

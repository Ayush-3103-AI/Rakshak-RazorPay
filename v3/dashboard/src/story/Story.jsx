// The front door: eight locked screens, a two-minute read.
//
// Gap → the product hole → the mechanism → the result → the stress test →
// what was killed → the lineage → verify. The problem first, in one breath;
// the number on screen four; the honesty after it, where a reader who has seen
// the claim goes looking for the catch.
//
// One gesture, one screen. The locking is CSS scroll-snap (tokens.css), not a
// wheel handler, so find-in-page, Page Down, Home/End and screen readers all
// keep working — the distinction that separates this from the scroll-jacking
// v1 was criticised for. The rail is the map: every screen visible at once,
// one click to any of them, and a lit spine showing how far in you are.
import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "framer-motion";
import {
  Activity,
  ArrowRight,
  GitBranch,
  Layers,
  ListChecks,
  ShieldAlert,
  Table2,
  TrendingUp,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import Brand from "../components/Brand.jsx";
import Rail, { RAIL_WIDTH, RailChips } from "../components/Rail.jsx";
import Gap from "./screens/Gap.jsx";
import Killed from "./screens/Killed.jsx";
import Lineage from "./screens/Lineage.jsx";
import Mechanism from "./screens/Mechanism.jsx";
import ProductHole from "./screens/ProductHole.jsx";
import Result from "./screens/Result.jsx";
import StressTest from "./screens/StressTest.jsx";
import Verify from "./screens/Verify.jsx";

export const STORY_SCREENS = [
  { id: "gap", label: "The gap", icon: ShieldAlert, group: "The problem" },
  { id: "product", label: "The product", icon: Layers, group: "The problem" },
  { id: "mechanism", label: "The mechanism", icon: Activity, group: "The problem" },
  { id: "result", label: "The result", icon: Table2, group: "The evidence" },
  { id: "stress", label: "The stress test", icon: TrendingUp, group: "The evidence" },
  { id: "killed", label: "What we killed", icon: ListChecks, group: "The discipline" },
  { id: "lineage", label: "The lineage", icon: GitBranch, group: "The discipline" },
  { id: "verify", label: "Verify", icon: ShieldAlert, group: "The discipline" },
];

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

/** The rail's foot: the read's length, and the way into the evidence panel. */
function RailFooter() {
  return (
    <div className="flex flex-col gap-[var(--spacing-4)]">
      <p className="m-0 font-mono text-[10px] tracking-[0.16em] text-faint uppercase">
        {STORY_SCREENS.length} screens · ~2 min
      </p>
      <a
        href="#/evidence"
        className="inline-flex items-center justify-between gap-[var(--spacing-3)] rounded-[var(--radius-md)] border border-primary/35 bg-primary/12 px-[var(--spacing-5)] py-[var(--spacing-4)] font-mono text-[11px] font-bold tracking-[0.14em] text-primary-text uppercase no-underline transition-colors duration-[var(--duration-quick)] hover:bg-primary/22"
      >
        Full evidence
        <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" />
      </a>
    </div>
  );
}

export default function Story() {
  const { scrollYProgress } = useScroll();
  const reduce = useReducedMotion();
  const bar = useSpring(scrollYProgress, { stiffness: 140, damping: 26, mass: 0.4 });
  const active = useActiveScreen();

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

      <Rail
        items={STORY_SCREENS}
        activeId={active}
        onSelect={go}
        footer={<RailFooter />}
        label="Story screens"
      />

      <RailChips
        items={STORY_SCREENS}
        activeId={active}
        onSelect={go}
        label="Story screens"
        leading={
          <>
            <Brand href="#/" descriptor={null} className="mr-[var(--spacing-3)] shrink-0" />
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

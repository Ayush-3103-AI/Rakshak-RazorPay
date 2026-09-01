// The shell: a left rail plus four ordinary scrolling sections. #61 asks for
// a shell that points at the v3 contract and hosts #62/#63/#64's content —
// not a re-implementation of ver1's scroll-jacked narrative pitch, which
// this evidence panel has no use for (see the dashboard report).
import { useEffect } from "react";
import Chrome from "./Chrome.jsx";
import { TooltipProvider } from "./components/ui/Tooltip.jsx";
import { readFragment, writeFragment } from "./fragment.js";
import DeferredRungs from "./sections/DeferredRungs.jsx";
import G5Figure from "./sections/G5Figure.jsx";
import Overview from "./sections/Overview.jsx";
import TrajectoryLadder from "./sections/TrajectoryLadder.jsx";
import { SECTIONS } from "./sections.js";
import { useScrolledSection } from "./useInView.js";

export default function App() {
  const activeSection = useScrolledSection();

  useEffect(() => {
    writeFragment(SECTIONS[activeSection].id);
  }, [activeSection]);

  useEffect(() => {
    const target = readFragment();
    if (target) document.getElementById(target)?.scrollIntoView({ block: "start" });
  }, []);

  return (
    <TooltipProvider>
      <Chrome activeSection={activeSection} />
      <main className="pl-[220px] max-md:pl-0">
        <section id="overview" className="scroll-mt-0">
          <Overview />
        </section>
        <section id="g5" className="scroll-mt-0">
          <G5Figure />
        </section>
        <section id="trajectory" className="scroll-mt-0">
          <TrajectoryLadder />
        </section>
        <section id="deferred" className="scroll-mt-0">
          <DeferredRungs />
        </section>
        <footer className="border-t border-border px-[var(--spacing-8)] py-[var(--spacing-7)] text-xs text-faint">
          RAKSHAK v3 — data-access layer reads <code className="font-mono">artifacts/*.json</code> only,
          no backend. Built for T-0127/#61, T-0128/#62, T-0129/#63, T-0130/#64.
        </footer>
      </main>
    </TooltipProvider>
  );
}

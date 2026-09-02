// The shell: a left rail plus six ordinary scrolling sections.
//
// The ordering is the argument (see sections.js for the reversal this encodes).
// §2 "How it is measured" sits BEFORE §3 "Results" on purpose: the reader this
// panel is written for — a risk operator who has seen a thousand backtests — has
// to be given grounds to trust the harness before being shown a number, or the
// number is just another vendor's chart.
//
// Still an ordinary document: no pinning, no playhead, no scroll-jacking. The
// fragment routing below is what makes a deep link to one result shareable, and
// nothing in the motion layer may break it.
import { useEffect } from "react";
import Chrome from "./Chrome.jsx";
import { TooltipProvider } from "./components/ui/Tooltip.jsx";
import { readFragment, writeFragment } from "./fragment.js";
import DeferredRungs from "./sections/DeferredRungs.jsx";
import Generations from "./sections/Generations.jsx";
import Method from "./sections/Method.jsx";
import Reproduce from "./sections/Reproduce.jsx";
import Results from "./sections/Results.jsx";
import Verdict from "./sections/Verdict.jsx";
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
        <section id="verdict" className="scroll-mt-0">
          <Verdict />
        </section>
        <section id="generations" className="scroll-mt-0">
          <Generations />
        </section>
        <section id="method" className="scroll-mt-0">
          <Method />
        </section>
        <section id="results" className="scroll-mt-0">
          <Results />
        </section>
        <section id="killed" className="scroll-mt-0">
          <DeferredRungs />
        </section>
        <section id="reproduce" className="scroll-mt-0">
          <Reproduce />
        </section>
        <footer className="border-t border-border px-[var(--spacing-8)] py-[var(--spacing-7)] text-xs text-faint">
          RAKSHAK G3 — data-access layer reads <code className="font-mono">artifacts/*.json</code> only,
          no backend. Built for T-0127/#61, T-0128/#62, T-0129/#63, T-0130/#64, and #79.
        </footer>
      </main>
    </TooltipProvider>
  );
}

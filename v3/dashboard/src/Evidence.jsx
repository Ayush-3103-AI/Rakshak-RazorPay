// The evidence panel: a collapsible rail, a status bar, and eight snap-paged
// screens. This was App.jsx until the story became the front door; nothing
// about its structure or its data changed in the move.
//
// PAGING IS CSS, NOT JAVASCRIPT. `scroll-snap-type: y mandatory` plus
// `scroll-snap-stop: always` on each page is what makes one gesture advance
// exactly one screen and never two — see tokens.css. Nothing here listens for
// `wheel`, nothing calls `preventDefault`, and nothing owns a playhead. That
// distinction is the whole reason this is acceptable where v1's scroll-jacking
// was not: browser find still lands on a match, Page Down and Home/End still
// work, a screen reader still walks the document in order, and the fragment
// deep-links still resolve. A reader hunting one number is never carried
// somewhere else — they are just parked neatly when they stop.
//
// Under `prefers-reduced-motion` the snap is dropped entirely (tokens.css) and
// this becomes an ordinary scrolling document again.
import { useCallback, useEffect, useState } from "react";
import Chrome, { RAIL_NARROW, RAIL_WIDE } from "./Chrome.jsx";
import { TooltipProvider } from "./components/ui/Tooltip.jsx";
import { readFragment, writeFragment } from "./fragment.js";
import ConfounderNull from "./sections/ConfounderNull.jsx";
import DeferredRungs from "./sections/DeferredRungs.jsx";
import Generations from "./sections/Generations.jsx";
import Ladder from "./sections/Ladder.jsx";
import Method from "./sections/Method.jsx";
import Reproduce from "./sections/Reproduce.jsx";
import Sweep from "./sections/Sweep.jsx";
import Verdict from "./sections/Verdict.jsx";
import { SECTIONS } from "./sections.js";
import { useScrolledSection } from "./useInView.js";

const COLLAPSE_KEY = "rakshak-v3-rail-collapsed";

export default function Evidence() {
  const activeSection = useScrolledSection();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
    } catch {
      /* private-mode storage can throw; the rail still toggles for this load */
    }
  }, [collapsed]);

  useEffect(() => {
    writeFragment(SECTIONS[activeSection]?.id);
  }, [activeSection]);

  useEffect(() => {
    const target = readFragment();
    if (target) document.getElementById(target)?.scrollIntoView({ block: "start" });
  }, []);

  const go = useCallback((id) => {
    document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, []);

  return (
    <TooltipProvider>
      <Chrome
        activeSection={activeSection}
        collapsed={collapsed}
        onToggleCollapsed={() => setCollapsed((c) => !c)}
        onSelect={go}
      />
      <main
        className="transition-[margin] duration-[var(--duration-moderate)] ease-[var(--ease-entrance)] max-md:!ml-0"
        style={{ marginLeft: collapsed ? RAIL_NARROW : RAIL_WIDE }}
      >
        <Verdict />
        <Generations />
        <Ladder />
        <Sweep />
        <ConfounderNull />
        <Method />
        <DeferredRungs />
        <Reproduce />
      </main>
    </TooltipProvider>
  );
}

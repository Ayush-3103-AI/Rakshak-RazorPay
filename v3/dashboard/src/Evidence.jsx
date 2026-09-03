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
import Chrome from "./Chrome.jsx";
import { RAIL_WIDTH } from "./components/Rail.jsx";
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

export default function Evidence() {
  const activeSection = useScrolledSection();

  // Captured during the FIRST RENDER, not in an effect. The effect below writes
  // the active section back into the hash, effects run in declaration order,
  // and it therefore used to overwrite `#/evidence/ladder` with
  // `#/evidence/verdict` before the reader ever ran — so every deep link
  // landed on section zero. A lazy state initialiser runs before any effect.
  const [initialTarget] = useState(readFragment);

  useEffect(() => {
    writeFragment(SECTIONS[activeSection]?.id);
  }, [activeSection]);

  useEffect(() => {
    if (initialTarget) document.getElementById(initialTarget)?.scrollIntoView({ block: "start" });
  }, [initialTarget]);

  const go = useCallback((id) => {
    document.getElementById(id)?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, []);

  return (
    <TooltipProvider>
      <Chrome activeSection={activeSection} onSelect={go} />
      <main style={{ marginRight: RAIL_WIDTH }} className="max-lg:!mr-0">
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

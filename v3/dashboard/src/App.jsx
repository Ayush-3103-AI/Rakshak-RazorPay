// The root is a two-way switch on the hash: the story (front door, a two-minute
// free-scrolling read) or the evidence panel (the eight snap-paged screens it
// used to open on). Both read the same committed artifacts through the same
// loader, so a number on the story and the same number on the panel cannot
// disagree — they are one fetch.
import { useEffect, useRef } from "react";
import Evidence from "./Evidence.jsx";
import { readFragment } from "./fragment.js";
import { useRoute } from "./route.js";
import Story from "./story/Story.jsx";

export default function App() {
  const route = useRoute();
  const previous = useRef(route);

  // Read DURING RENDER, not inside the effect. The panel's own effect writes
  // the section it is showing back into the hash, and child effects run before
  // the parent's — so by the time this component's effect runs, a plain
  // `#/evidence` has already become `#/evidence/verdict`. Reading it there made
  // the reset below think the reader had asked for a section and skip, which is
  // why arriving from the story left you wherever you had been standing in the
  // story rather than at the top of the panel. Render happens before any of
  // those effects, so what we capture here is what the reader actually asked
  // for. (`replaceState` fires no hashchange, so this does not re-run.)
  const entryFragment = readFragment();

  useEffect(() => {
    if (previous.current === route) return;
    previous.current = route;
    // Only a real route CHANGE resets to the top, never the first mount — and
    // never when the incoming hash names a section, because the panel has
    // already scrolled to that target and an unconditional reset would undo it.
    if (!entryFragment) window.scrollTo(0, 0);
  }, [route, entryFragment]);

  return route === "evidence" ? <Evidence /> : <Story />;
}

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

  useEffect(() => {
    if (previous.current === route) return;
    previous.current = route;
    // Only a real route CHANGE resets to the top, never the first mount — and
    // never when the incoming hash names a section. Child effects run BEFORE
    // the parent's, so the panel has already scrolled to its deep-link target
    // by the time this runs, and an unconditional reset here would undo it:
    // `#/evidence/ladder` would land on the ladder and then be yanked back to
    // the first screen.
    if (!readFragment()) window.scrollTo(0, 0);
  }, [route]);

  return route === "evidence" ? <Evidence /> : <Story />;
}

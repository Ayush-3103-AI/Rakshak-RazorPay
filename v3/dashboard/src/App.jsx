// The root is a two-way switch on the hash: the story (front door, a two-minute
// free-scrolling read) or the evidence panel (the eight snap-paged screens it
// used to open on). Both read the same committed artifacts through the same
// loader, so a number on the story and the same number on the panel cannot
// disagree — they are one fetch.
import { useEffect, useRef } from "react";
import Evidence from "./Evidence.jsx";
import { useRoute } from "./route.js";
import Story from "./story/Story.jsx";

export default function App() {
  const route = useRoute();
  const previous = useRef(route);

  useEffect(() => {
    // tokens.css keys the snap paging on this attribute, so the story scrolls
    // freely and the panel pages.
    document.documentElement.setAttribute("data-route", route);
    // Child effects run first, so the panel has already honoured a section
    // fragment by the time this runs; only a real route CHANGE resets to the
    // top, never the first mount.
    if (previous.current !== route) {
      previous.current = route;
      window.scrollTo(0, 0);
    }
  }, [route]);

  return route === "evidence" ? <Evidence /> : <Story />;
}

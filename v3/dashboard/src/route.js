// Two routes, one hash. The story is the front door and the evidence panel
// sits behind `#/evidence`; the section deep-links the panel already had
// become `#/evidence/<section>` (see fragment.js).
//
// A hash, not a path, because GitHub Pages serves a single index.html and has
// no rewrite rules: a real `/evidence` URL would 404 on a cold load. No router
// dependency either — this is one listener and one string test.
import { useEffect, useState } from "react";

export const EVIDENCE_PREFIX = "#/evidence";

export function routeFromHash(hash = globalThis.location?.hash ?? "") {
  return hash.startsWith(EVIDENCE_PREFIX) ? "evidence" : "story";
}

export function useRoute() {
  const [route, setRoute] = useState(() => routeFromHash());
  useEffect(() => {
    const onChange = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

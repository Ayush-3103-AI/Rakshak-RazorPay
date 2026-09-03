// The URL fragment carries section identity inside the evidence route, so any
// part of the panel is deep-linkable and restorable on load:
// `#/evidence/ladder`. Ported from ver1/dashboard/src/fragment.js (T-0127,
// #61), trimmed of the tab segment v1 needed, and now prefixed with the route
// so the story can own the bare hash.
import { EVIDENCE_PREFIX } from "./route.js";
import { SECTIONS } from "./sections.js";

export function readFragment() {
  const hash = globalThis.location?.hash ?? "";
  if (!hash.startsWith(`${EVIDENCE_PREFIX}/`)) return null;
  const id = hash.slice(EVIDENCE_PREFIX.length + 1);
  return SECTIONS.some((s) => s.id === id) ? id : null;
}

/** replaceState, not a hash assignment: scrolling the page is not a history event. */
export function writeFragment(id) {
  if (!id) return;
  const hash = `${EVIDENCE_PREFIX}/${id}`;
  if (globalThis.location?.hash !== hash) {
    globalThis.history?.replaceState(null, "", hash);
  }
}

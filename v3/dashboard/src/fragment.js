// The URL fragment carries section identity, so any part of the panel is
// deep-linkable and restorable on load: `#g5`. Ported from
// ver1/dashboard/src/fragment.js (T-0127, #61), trimmed of the tab segment
// v1 needed for its "Under the hood" tab strip — this shell has no tabs.
import { SECTIONS } from "./sections.js";

export function readFragment() {
  const id = (globalThis.location?.hash ?? "").replace(/^#/, "");
  return SECTIONS.some((s) => s.id === id) ? id : null;
}

/** replaceState, not a hash assignment: scrolling the page is not a history event. */
export function writeFragment(id) {
  const hash = `#${id}`;
  if (globalThis.location?.hash !== hash) {
    globalThis.history?.replaceState(null, "", hash);
  }
}

// The page manifest — one source of truth for the rail, the URL fragment and
// the snap order.
//
// DECISION AMENDED. Two things changed from the #79 shell and both are
// deliberate.
//
// 1. RESULTS COME FIRST. #79 put §2 "How it is measured" before §3 "Results" on
//    the argument that a risk operator needs grounds to trust the harness before
//    being shown a number. That argument was right about a DOCUMENT and wrong
//    about a PANEL. A judge with six submissions to get through does not read a
//    preamble; they look for the number, and if it is not on the first screen
//    they decide the project does not have one. The trust argument still runs —
//    it is now the page immediately after the evidence, where a reader who has
//    seen the claim goes looking for the catch. Same argument, met at the moment
//    the reader actually wants it.
//
// 2. ONE SECTION IS ONE PAGE. §3 used to render the ladder, the sweep and the
//    G5 null in a single scrolling section. Under the snap model each of those
//    is a screen of its own, so they are three entries here. Nothing was cut in
//    the split — the ladder, the sweep, the decomposition and the null are all
//    still on the panel, one figure per screen instead of three stacked.
//
// What still stands from #79: no pinning, no playhead, no wheel-jacking. The
// paging is CSS scroll-snap, so browser find still works, Page Down still works,
// and the fragment deep-links still resolve. See App.jsx.
export const SECTIONS = [
  { id: "verdict", label: "Verdict", eyebrow: "§0", group: "Overview" },
  { id: "lineage", label: "Lineage", eyebrow: "§1", group: "Overview" },
  { id: "ladder", label: "The ladder", eyebrow: "§2", group: "Evidence" },
  { id: "sweep", label: "Cost sweep", eyebrow: "§3", group: "Evidence" },
  { id: "null", label: "Confounder null", eyebrow: "§4", group: "Evidence" },
  { id: "method", label: "How it's measured", eyebrow: "§5", group: "Discipline" },
  { id: "killed", label: "What we killed", eyebrow: "§6", group: "Discipline" },
  { id: "reproduce", label: "Reproduce", eyebrow: "§7", group: "Provenance" },
];

/** The rail renders by group; this keeps the grouping derived, never duplicated. */
export const SECTION_GROUPS = SECTIONS.reduce((groups, section, index) => {
  const last = groups[groups.length - 1];
  if (last && last.name === section.group) last.items.push({ ...section, index });
  else groups.push({ name: section.group, items: [{ ...section, index }] });
  return groups;
}, []);

// The section manifest — one source of truth for the rail and the URL fragment.
//
// DECISION REVERSED (#79). This file previously read: "Unlike ver1's scroll-jacked
// narrative pitch, this is a judge-facing evidence panel: an ordinary scrolling
// document, four sections, no pinning." Half of that still holds and half of it was
// wrong, so it is corrected here rather than left standing against the code.
//
// What stands: no pinning, no playhead, no scroll-jacking. Browser find works, the
// fragment deep-links resolve, and a reader hunting one number is never carried
// somewhere else. #61's argument against porting v1's jacking machinery was right and
// nothing below re-introduces it.
//
// What was wrong: that a panel of evidence with no argument around it is the safer
// choice. It is not. The four sections presented measurements to a reader who had been
// given no reason to trust the harness that produced them, and left this project's
// actual differentiator — pre-registration, four sealed locks, a test split kept shut
// twice, six negative results reported with numbers — legible only to someone who
// read the repository. So the shell now argues the method BEFORE it shows a number,
// which is the whole of the reordering: §2 precedes §3 on purpose.
//
// Every previously existing section survives. None was deleted; three were re-homed.
export const SECTIONS = [
  { id: "verdict", label: "Verdict", eyebrow: "§0" },
  { id: "generations", label: "The gap & three generations", eyebrow: "§1" },
  { id: "method", label: "How it is measured", eyebrow: "§2" },
  { id: "results", label: "Results", eyebrow: "§3" },
  { id: "killed", label: "What we killed", eyebrow: "§4" },
  { id: "reproduce", label: "Reproduce", eyebrow: "§5" },
];

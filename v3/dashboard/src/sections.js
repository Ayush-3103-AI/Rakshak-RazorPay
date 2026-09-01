// The section manifest — one source of truth for the rail and the URL fragment.
// Unlike ver1's scroll-jacked narrative pitch, this is a judge-facing evidence
// panel: an ordinary scrolling document, four sections, no pinning. #61 asks
// for a shell that points at the new contract and hosts #62/#63/#64's content;
// the jacking/playhead machinery that drove v1's landing-page narrative is not
// part of that job, so it was not ported (see the dashboard report for why).
export const SECTIONS = [
  { id: "overview", label: "Overview", eyebrow: "§0" },
  { id: "g5", label: "G5 — confounder null", eyebrow: "§1" },
  { id: "trajectory", label: "Trajectory & ladder", eyebrow: "§2" },
  { id: "deferred", label: "Rungs 5–8", eyebrow: "§3" },
];

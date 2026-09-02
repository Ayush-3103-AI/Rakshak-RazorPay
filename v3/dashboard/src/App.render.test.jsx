// One end-to-end check: mount the whole panel against the REAL committed
// artefacts on disk and assert every section rendered its data rather than
// its error state. This is the test that would have caught a crash on a null
// metric, a non-finite census, or a window shading off the end of the axis —
// all of which are present in the real files.
//
// Extended for #79: six sections rather than four, the two new artefacts, and
// the external "cited, not recomputed" marker. Fixtures are still refused on
// purpose — a test that passes against invented data cannot tell you the panel
// survives the files you are about to publish.
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { clearArtifactCache } from "./lib/artifacts.js";

const DIR = join(process.cwd(), "public", "artifacts");

// `reduceMotion` drives the one media query framer-motion reads. The panel must
// render the same INFORMATION either way — only the transition between states
// differs — so both paths run the same assertions.
function stubEnvironment({ reduceMotion = false } = {}) {
  clearArtifactCache();
  // ResizeObserver is what recharts' ResponsiveContainer needs; jsdom has none.
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  globalThis.matchMedia = (query) => ({
    matches: reduceMotion && String(query).includes("prefers-reduced-motion"),
    media: String(query),
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  });
  // jsdom implements neither of these; both are browser-only affordances.
  Element.prototype.scrollIntoView = () => {};
  globalThis.IntersectionObserver = class {
    constructor(cb) {
      this.cb = cb;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  vi.stubGlobal("fetch", async (url) => {
    const name = String(url).split("/").pop();
    try {
      const body = readFileSync(join(DIR, name), "utf-8");
      return { ok: true, status: 200, json: async () => JSON.parse(body) };
    } catch {
      return { ok: false, status: 404 };
    }
  });
}

beforeEach(() => stubEnvironment());
afterEach(() => {
  // Explicit, because this project does not run vitest with `globals: true` —
  // which is what @testing-library/react's auto-cleanup hooks itself onto. Without
  // it the second test mounts a second <App /> beside the first and every
  // single-element query in the file starts failing with "found multiple".
  cleanup();
  vi.unstubAllGlobals();
});

it("renders every section from the committed artefacts without falling back to an error state", async () => {
  render(<App />);

  // Every artifact is fetched in parallel from the same stub, so ONE wait covers
  // the lot. Waiting per section instead multiplies the timeout budget by six for
  // no extra coverage — and that is what pushed this file past the default.
  await waitFor(() => expect(screen.getByText(/Specified, gated, and either/)).toBeTruthy());

  // §0 — the verdict, computed from the artefacts rather than typed into the hero.
  expect(screen.getByText(/policies on the ladder/)).toBeTruthy();
  expect(screen.getByText(/sealed eval locks/)).toBeTruthy();
  expect(screen.getByText(/rows? beat every floor/)).toBeTruthy();
  // The disclosure that governs how every figure below it must be read.
  expect(screen.getByText(/BAF is not vendored in this tree/)).toBeTruthy();

  // §1 — the three generations, from journey.json, with G1 marked as foreign.
  expect(screen.getByText(/Nothing watches a merchant/)).toBeTruthy();
  expect(screen.getByText("G1")).toBeTruthy();
  expect(screen.getByText("G3")).toBeTruthy();
  expect(screen.getAllByText(/cited, not recomputed/).length).toBeGreaterThan(0);

  // §2 — the method, argued before any number is shown.
  expect(screen.getByText(/Read this before you read a single number/)).toBeTruthy();
  expect(screen.getAllByText("eval_module_sha256").length).toBeGreaterThan(0);
  expect(screen.getByText(/The floors, named/)).toBeTruthy();

  // §3a — the ladder rendered rows and named the floor each one loses to.
  expect(screen.getByText(/Every policy, against the same floors/)).toBeTruthy();
  expect(screen.getAllByText(/under volume_rank/).length).toBeGreaterThan(0);

  // §3b — the sweep, its operating point, and the decomposition beneath it.
  expect(screen.getByText(/Does the ranking survive a different cost matrix/)).toBeTruthy();
  // The operating point, from the artifact's own `shipped_ratio_within_grid` —
  // matched on "shipped ratio" rather than "inside the grid", which G3's journey
  // note also contains.
  expect(screen.getByText(/shipped ratio/)).toBeTruthy();
  expect(screen.getByText(/Where the margin actually comes from/)).toBeTruthy();
  expect(screen.getByText(/HOLD forbidden/)).toBeTruthy();

  // §3c — G5 read its verdicts and shaded from `role`, not a hardcoded list.
  expect(screen.getByText(/Alert rate by simulation day/)).toBeTruthy();
  expect(screen.getAllByText(/adversarial/).length).toBeGreaterThan(0);

  // §4 — the roster drove the killed section. The old assertion here was on
  // UNVERIFIED, which the roster no longer carries (`n_unverified` is 0 since the
  // lead confirmed every entry); asserting on a status that has since been
  // resolved would fail for a good reason and teach nothing. What must hold is
  // that the section is roster-driven at all, so it asserts on the statuses the
  // committed roster does carry.
  expect(screen.getByText(/Specified, gated, and either unscored or not adopted/)).toBeTruthy();
  expect(screen.getAllByText("scored").length).toBeGreaterThan(0);
  expect(screen.getAllByText("cut").length).toBeGreaterThan(0);

  // §5 — the manifest's own PRESENT/MISSING account, including the new artefacts.
  expect(screen.getAllByText("PRESENT").length).toBeGreaterThan(0);
  expect(screen.getByText("cost_sweep")).toBeTruthy();
  expect(screen.getByText("journey")).toBeTruthy();

  // The invariant that outranks all of them: no TEST-split anything, anywhere.
  expect(screen.queryByText("TEST")).toBeNull();

  // And no artefact fell back to the fail-visible error card.
  expect(screen.queryByText(/MISSING or invalid/)).toBeNull();
});

it("renders the same figures under prefers-reduced-motion, with no counter left at zero", async () => {
  vi.unstubAllGlobals();
  stubEnvironment({ reduceMotion: true });
  render(<App />);

  await waitFor(() => expect(screen.getByText(/Read this before you read a single number/)).toBeTruthy());
  expect(screen.getByText(/policies on the ladder/)).toBeTruthy();

  // Counters snap to the artefact's value rather than animating toward it. The
  // ladder has rows, so a "0" in the hero tiles would mean the tween never ran
  // — which is exactly the failure a reduced-motion reader would be left with.
  const tiles = document.querySelectorAll(".tabular-nums");
  expect(tiles.length).toBeGreaterThan(0);
  expect([...tiles].some((el) => el.textContent.trim() !== "0")).toBe(true);

  // Every section still renders its data, not an error card.
  expect(screen.getByText(/Read this before you read a single number/)).toBeTruthy();
  expect(screen.queryByText(/MISSING or invalid/)).toBeNull();
});

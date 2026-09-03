// The reduced-motion path, in a FILE OF ITS OWN — and that is the whole point.
//
// framer-motion latches its reduced-motion answer in module scope the first time
// `useReducedMotion` runs, and never re-reads `matchMedia` afterwards. A second
// test inside App.render.test.jsx therefore cannot switch the preference: the
// first test has already pinned it to "no preference", the stub is ignored, and
// the test passes while asserting nothing. Vitest isolates module state per file,
// so a separate file is what actually exercises the branch.
//
// What must hold: a reader with reduced motion set sees the SAME NUMBERS, not an
// empty page and not a counter frozen at its starting value. The counters are
// checked against the committed artifacts rather than against literals, so this
// keeps testing the real contract after a rescore changes every figure.
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { clearArtifactCache } from "./lib/artifacts.js";

const DIR = join(process.cwd(), "public", "artifacts");
const read = (name) => JSON.parse(readFileSync(join(DIR, name), "utf-8"));

beforeEach(() => {
  clearArtifactCache();
  // The evidence panel lives behind the hash route; the four counters this test
  // walks are its verdict tiles, not the story's.
  window.location.hash = "#/evidence";
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  // The one query framer-motion reads, answered "reduce" — before any render.
  globalThis.matchMedia = (query) => ({
    matches: String(query).includes("prefers-reduced-motion"),
    media: String(query),
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  });
  Element.prototype.scrollIntoView = () => {};
  globalThis.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  vi.stubGlobal("fetch", async (url) => {
    const name = String(url).split("/").pop();
    try {
      return { ok: true, status: 200, json: async () => read(name) };
    } catch {
      return { ok: false, status: 404 };
    }
  });
});

afterEach(() => {
  window.location.hash = "";
  cleanup();
  vi.unstubAllGlobals();
});

it("shows every counter at its artifact value under prefers-reduced-motion", async () => {
  const ladder = read("ladder.json").payload;
  const lockState = read("lock_state.json").payload;

  const seeds = ladder.rungs.map((r) => r.n_seeds ?? 0);
  // DOM order, and the DOM order is the editorial order: the survivor count is
  // the headline of the whole panel, so it is the first tile and the only filled
  // one. If a future edit demotes it back into the middle of the row, this
  // assertion should fail and be argued with rather than resorted.
  const expected = [
    ladder.rungs.filter((r) => r.beats_all_floors).length,
    ladder.rungs.length,
    Math.min(...seeds),
    lockState.locks.length,
  ].map((n) => n.toLocaleString("en-IN"));

  render(<App />);
  // Wait on something only the ARTIFACTS can produce. The tile labels are static
  // copy and render at frame zero with every counter still at its initial value,
  // so waiting on those samples the page before the data has arrived and reads
  // four zeroes that mean "not loaded yet", not "reduced motion is broken".
  //
  // BOTH artefacts, not just the lock. Three of the four counters are derived
  // from ladder.json and only the fourth from lock_state.json, so gating on the
  // lock hash alone lets the assertion run while the ladder is still in flight.
  // That raced: it passed whenever the two fetches resolved in the same tick and
  // failed under load, which is the worst failure mode a test can have. The
  // ladder's own rendered output is the honest gate.
  await waitFor(() => {
    expect(screen.getAllByText("eval_module_sha256").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/under volume_rank/).length).toBeGreaterThan(0);
  });

  // `[data-counter]`, not `.tabular-nums` — the latter matches every mono figure
  // on the page, which is how this assertion used to pass on a journey literal
  // while all four counters sat at "0".
  const counters = [...document.querySelectorAll("[data-counter]")].map((el) => el.textContent.trim());
  expect(counters).toEqual(expected);

  // A counter still at its starting value is precisely the failure a reduced-motion
  // reader would be left with, so name it rather than relying on the deep-equal above.
  expect(counters.filter((t) => t === "0")).toEqual(
    expected.filter((t) => t === "0"),
    "a counter is stuck at 0 that the artifacts do not say is 0"
  );

  expect(screen.queryByText(/MISSING or invalid/)).toBeNull();
});

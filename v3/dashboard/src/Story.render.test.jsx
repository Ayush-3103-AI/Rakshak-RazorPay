// The front door, mounted against the REAL committed artefacts: every screen
// that states a number derives it, so this asserts on numbers computed from
// the files rather than on literals — the headline count, the sweep band, the
// lock chip — and on the invariants the evidence panel already enforces.
// Fixtures are refused on purpose, as in App.render.test.jsx.
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { clearArtifactCache } from "./lib/artifacts.js";
import { fmtNum } from "./lib/format.js";
import { deriveLadder, deriveSweep } from "./story/derive.js";

const DIR = join(process.cwd(), "public", "artifacts");
const read = (name) => JSON.parse(readFileSync(join(DIR, name), "utf-8"));

beforeEach(() => {
  clearArtifactCache();
  window.location.hash = "";
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  // No reduced motion and no wide viewport: the pinned screens take their
  // static path here, which is the path a phone reader gets.
  globalThis.matchMedia = (query) => ({
    matches: false,
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
  cleanup();
  vi.unstubAllGlobals();
});

it("renders the story from the committed artefacts, every number derived", async () => {
  const ladder = deriveLadder(read("ladder.json"));
  const sweep = deriveSweep(read("cost_sweep.json"), ladder.survivorLabel);
  const locks = read("lock_state.json").payload.locks;
  const opens = locks.reduce((n, l) => n + (l.open_count ?? 0), 0);

  render(<App />);

  await waitFor(() => {
    // §4 the result — the headline sentence the ladder completes
    expect(screen.getByText(new RegExp(`^${ladder.rungs.length} policies raced`))).toBeTruthy();
    // §1 the lock chip, from lock_state
    expect(screen.getByText(new RegExp(`${locks.length} sealed locks .* test split opened ${opens}×`))).toBeTruthy();
    // §7 journey rendered, with G1 marked foreign
    expect(screen.getAllByText(/cited, not recomputed/).length).toBeGreaterThan(0);
    // §6 the roster drove the failures screen
    expect(screen.getAllByText("not adopted").length).toBeGreaterThan(0);
    // §8 the manifest's names are on the page
    expect(screen.getByText("cost_sweep")).toBeTruthy();
  });

  // §5 the band in the lede is the sweep's own min and max, formatted the same way
  if (sweep?.band) {
    expect(screen.getByText(new RegExp(`${fmtNum(sweep.band[0], 4)} and ${fmtNum(sweep.band[1], 4)}`))).toBeTruthy();
  }
  if (sweep?.decisionShare != null) {
    expect(screen.getByText(`${Math.round(sweep.decisionShare * 100)}%`)).toBeTruthy();
  }

  // The survivor is named, on the result screen and in the bars' label.
  if (ladder.survivorLabel) {
    expect(screen.getAllByText(ladder.survivorLabel).length).toBeGreaterThan(0);
  }

  // The invariants the panel enforces hold on the front door too.
  expect(screen.queryByText("TEST")).toBeNull();
  expect(screen.queryByText(/MISSING or invalid/)).toBeNull();
});

it("switches to the evidence panel on its hash route", async () => {
  window.location.hash = "#/evidence";
  const { container } = render(<App />);
  await waitFor(() => {
    expect(screen.getAllByText("eval_module_sha256").length).toBeGreaterThan(0);
  });
  expect(screen.queryByText(/policies raced/)).toBeNull();
  expect(screen.getAllByText(/Back to the story/).length).toBeGreaterThan(0);
  // The panel's four verdict tiles, and only those: the story's counters must
  // not be in the tree at all on this route.
  expect(container.querySelectorAll("[data-counter]").length).toBe(4);
});

// One end-to-end check: mount the whole panel against the REAL committed
// artefacts on disk and assert every section rendered its data rather than
// its error state. This is the test that would have caught a crash on a null
// metric, a non-finite census, or a window shading off the end of the axis —
// all of which are present in the real files.
import { render, screen, waitFor } from "@testing-library/react";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { clearArtifactCache } from "./lib/artifacts.js";

const DIR = join(process.cwd(), "public", "artifacts");

beforeEach(() => {
  clearArtifactCache();
  // ResizeObserver is what recharts' ResponsiveContainer needs; jsdom has none.
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  // jsdom implements neither of these; both are browser-only affordances.
  Element.prototype.scrollIntoView = () => {};
  globalThis.IntersectionObserver = class {
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
});

afterEach(() => vi.unstubAllGlobals());

it("renders every section from the committed artefacts without falling back to an error state", async () => {
  render(<App />);

  // §0 — the manifest's own account of each artefact.
  await waitFor(() => expect(screen.getAllByText("PRESENT").length).toBeGreaterThan(0));
  expect(screen.getByText(/RAKSHAK v3 — the evidence panel/)).toBeTruthy();

  // §1 — G5 read its verdicts and shaded from `role`, not a hardcoded list.
  await waitFor(() => expect(screen.getByText(/Alert rate by simulation day/)).toBeTruthy());
  expect(screen.getAllByText(/adversarial/).length).toBeGreaterThan(0);

  // §2 — the ladder rendered rows and the lock rendered its hashes.
  await waitFor(() => expect(screen.getByText("The model ladder")).toBeTruthy());
  expect(screen.getAllByText("eval_module_sha256").length).toBeGreaterThan(0);
  expect(screen.getAllByText(/under volume_rank/).length).toBeGreaterThan(0);

  // §3 — the roster drove the deferred section, UNVERIFIED included.
  await waitFor(() => expect(screen.getByText(/Specified, gated, and deliberately unscored/)).toBeTruthy());
  expect(screen.getAllByText("UNVERIFIED").length).toBeGreaterThan(0);

  // The invariant that outranks all of them: no TEST-split anything, anywhere.
  expect(screen.queryByText("TEST")).toBeNull();

  // And no artefact fell back to the fail-visible error card.
  expect(screen.queryByText(/MISSING or invalid/)).toBeNull();
});

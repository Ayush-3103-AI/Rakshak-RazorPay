// The arithmetic behind every number the story states, checked two ways: on
// the committed artefacts (so the front door and the files agree) and on
// small documents that exercise the branches the real files happen not to.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, it } from "vitest";
import { deriveKilled, deriveLadder, deriveSweep } from "./derive.js";

const DIR = join(process.cwd(), "public", "artifacts");
const read = (name) => JSON.parse(readFileSync(join(DIR, name), "utf-8"));

it("deriveLadder: survivors, seeds and rows agree with the committed ladder", () => {
  const doc = read("ladder.json");
  const d = deriveLadder(doc);
  const rungs = doc.payload.rungs;
  expect(d.rungs.length).toBe(rungs.length);
  expect(d.survivors.map((r) => r.label)).toEqual(rungs.filter((r) => r.beats_all_floors).map((r) => r.label));
  expect(d.seedCount).toBe(Math.min(...rungs.map((r) => r.n_seeds)));
  // sorted by margin, descending, and every margin is savings minus that row's own floor
  for (let i = 1; i < d.rows.length; i += 1) expect(d.rows[i - 1].margin).toBeGreaterThanOrEqual(d.rows[i].margin);
  for (const row of d.rows) {
    const src = rungs.find((r) => r.label === row.label);
    expect(row.margin).toBeCloseTo(src.metrics.savings - src.metrics.savings_floor_volume_rank, 12);
  }
});

it("deriveSweep: band, span and the decision-layer share come from the sweep's own fields", () => {
  const ladder = deriveLadder(read("ladder.json"));
  const doc = read("cost_sweep.json");
  const s = deriveSweep(doc, ladder.survivorLabel);
  const p = doc.payload;
  expect(s.policy).toBe(ladder.survivorLabel.replace(/_realised_exposure$/, ""));
  expect(s.band).toEqual([Math.min(...s.series.values), Math.max(...s.series.values)]);
  expect(s.span).toBeCloseTo(Math.max(...p.ratios) / Math.min(...p.ratios), 9);
  const idx = p.ratios.indexOf(p.hold_decomposition_at_ratio);
  const hd = p.hold_decomposition.find((h) => h.policy === s.policy);
  const floorAt = p.arms.review_only.find((a) => a.policy === "volume_rank").values[idx];
  expect(s.decisionShare).toBeCloseTo(hd.delta / (hd.with_hold - floorAt), 12);
});

it("deriveSweep: a missing floor or decomposition drops the number rather than inventing one", () => {
  const doc = {
    split: "VALIDATION",
    payload: { ratios: [1, 10], arms: { realised: [{ policy: "rung4", values: [0.5, 0.6] }] } },
  };
  const s = deriveSweep(doc, "rung4_realised_exposure");
  expect(s.band).toEqual([0.5, 0.6]);
  expect(s.floor).toBeNull();
  expect(s.beatsAt).toBeNull();
  expect(s.decisionShare).toBeNull();
});

it("deriveKilled: takes the roster's not-adopted and cut entries, strips amendment prefixes, and adds the null only when it fires", () => {
  const roster = {
    payload: {
      roster: [
        { rung: 1, name: "keep", title: "kept", status: "scored", adopted: null },
        { rung: 7, name: "drop", title: "dropped", status: "built", adopted: false, amended: "2026-09-02, T-0124. planned -> built, adopted -> false. Measured and it LOSES." },
        { rung: 6, name: "cut", title: "cut one", status: "cut", adopted: null, reason: "Cut. No code." },
        { rung: 5, name: "torch", title: "torch one", status: "built", adopted: false, reason: "Needs autograd.", result: "NOT ADOPTED, by a margin." },
      ],
    },
  };
  const green = { payload: { excess_allowed_pp: 2, series: [{ detector: "raw", verdict: "GREEN", window_excess: [] }] } };
  const red = {
    payload: {
      excess_allowed_pp: 2,
      series: [
        { detector: "raw", verdict: "RED", window_excess: [{ role: "adversarial", confounder: "P1", start_day: 93, end_day: 98, excess_pp: 7.07 }] },
        { detector: "cohort-residual", verdict: "RED", window_excess: [{ role: "adversarial", confounder: "P1", start_day: 93, end_day: 98, excess_pp: 2.7 }] },
      ],
    },
  };
  const onGreen = deriveKilled(roster, green);
  expect(onGreen.map((e) => e.name)).toEqual(["drop", "cut", "torch"]);
  expect(onGreen[0].verdict).toBe("Measured and it LOSES.");
  expect(onGreen[2].verdict).toBe("NOT ADOPTED, by a margin."); // result outranks the stale blocker
  const onRed = deriveKilled(roster, red);
  expect(onRed.at(-1).name).toBe("confounder_null");
  expect(onRed.at(-1).verdict).toMatch(/RED on every detector \(2\)/);
  expect(onRed.at(-1).verdict).toMatch(/7\.07 pp above nominal, against an allowed 2 pp/);
});

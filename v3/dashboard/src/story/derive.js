// Every figure the story states, derived from the artifact documents — pure
// functions, no React, so the arithmetic that puts a number into a headline is
// the same arithmetic a test can run against the committed files.
//
// Nothing in here holds a value. A number that is not in the artifacts comes
// out as null, and the screens render the sentence without it rather than
// with a guess.
import { asText } from "../lib/format.js";

export function survivorArm(label) {
  return label?.endsWith("_realised_exposure") ? "realised" : "declared";
}

export function survivorPolicy(label) {
  return label?.replace(/_realised_exposure$/, "") ?? null;
}

export function deriveLadder(doc) {
  const rungs = doc?.payload?.rungs ?? [];
  const survivors = rungs.filter((r) => r.beats_all_floors);
  // The MINIMUM seed count, as the panel does: "5 seeds" must be true of every
  // row it is said of.
  const seedCounts = rungs.map((r) => r.n_seeds ?? 0);
  const seedCount = seedCounts.length ? Math.min(...seedCounts) : 0;
  const seedsUniform = seedCounts.length > 0 && Math.max(...seedCounts) === seedCount;
  const floors = (doc?.payload?.metric_keys ?? []).filter((k) => k.startsWith("savings_floor_"));

  // Margin over the hardest floor, per row. volume_rank is the floor that beats
  // every rung on raw ranking, so "beats every floor" and "positive margin over
  // volume_rank" coincide on this ladder; the bars show the second, the badge
  // shows the first, and they are read from different fields on purpose.
  const rows = rungs
    .map((r) => {
      const savings = r.metrics?.savings;
      const floor = r.metrics?.savings_floor_volume_rank;
      const margin = Number.isFinite(savings) && Number.isFinite(floor) ? savings - floor : null;
      return {
        label: r.label,
        rung: r.rung,
        savings: Number.isFinite(savings) ? savings : null,
        floor: Number.isFinite(floor) ? floor : null,
        margin,
        beats: Boolean(r.beats_all_floors),
        floorFail: r.floor_fail ?? [],
        nSeeds: r.n_seeds ?? 0,
      };
    })
    .sort((a, b) => (b.margin ?? -Infinity) - (a.margin ?? -Infinity));

  const headline = survivors[0] ?? rungs[0] ?? null;
  return {
    rungs,
    survivors,
    survivorLabel: survivors[0]?.label ?? null,
    seedCount,
    seedsUniform,
    floors,
    rows,
    split: doc?.split ?? null,
    prevalence: headline?.metrics?.prevalence ?? null,
    capacityK: headline?.metrics?.capacity_k ?? null,
  };
}

export function deriveSweep(doc, survivorLabel) {
  const p = doc?.payload;
  if (!p) return null;
  const ratios = p.ratios ?? [];
  const arm = survivorArm(survivorLabel);
  const policy = survivorPolicy(survivorLabel);
  const armSeries = p.arms?.[arm] ?? [];
  const series = armSeries.find((s) => s.policy === policy) ?? null;
  const others = armSeries.filter((s) => s.policy !== policy);
  const floor = (p.arms?.review_only ?? []).find((s) => s.policy === "volume_rank") ?? null;
  const band = series?.values?.length ? [Math.min(...series.values), Math.max(...series.values)] : null;
  const span = ratios.length ? Math.max(...ratios) / Math.min(...ratios) : null;
  const beatsAt =
    series && floor ? series.values.filter((v, i) => Number.isFinite(floor.values[i]) && v > floor.values[i]).length : null;

  // What HOLD is worth: the decomposition's delta as a share of the margin over
  // the floor, both read at the ratio the decomposition was taken at.
  const idx = ratios.indexOf(p.hold_decomposition_at_ratio);
  const hold = (p.hold_decomposition ?? []).find((h) => h.policy === policy) ?? null;
  let decisionShare = null;
  if (hold && floor && idx >= 0 && Number.isFinite(floor.values[idx])) {
    const margin = hold.with_hold - floor.values[idx];
    if (margin > 0 && Number.isFinite(hold.delta)) decisionShare = hold.delta / margin;
  }

  return {
    ratios,
    arm,
    policy,
    series,
    others,
    floor,
    band,
    span,
    beatsAt,
    hold,
    decisionShare,
    shippedRatio: p.shipped_ratio ?? null,
    shippedWithin: Boolean(p.shipped_ratio_within_grid),
    meta: p.meta ?? {},
    split: doc.split ?? null,
  };
}

// The roster's fields are prose written at the time of each verdict. The most
// telling one differs per entry: `result` carries the measured verdict where
// there is one, `adoption_note` where the gate was argued, `amended` where the
// entry was rewritten after scoring. `reason` LAST — for the two torch-gated
// rungs it records the blocker that was later reversed, not the verdict.
function verdictText(entry) {
  const raw = entry.result ?? entry.adoption_note ?? entry.amended ?? entry.reason ?? entry.note ?? "";
  // Amendment entries open with a date, a ticket and a status-change clause
  // ("2026-09-02, T-0124. planned -> built, adopted -> false. ") before the
  // sentence that matters; strip exactly that shape and nothing else.
  return asText(raw).replace(/^\d{4}-\d{2}-\d{2},\s*[^.]*\.\s*(?:[a-z ]+->[^.]*\.\s*)?/, "");
}

export function deriveKilled(rosterDoc, g5Doc) {
  const roster = rosterDoc?.payload?.roster ?? [];
  const entries = roster
    .filter((e) => e.adopted === false || e.status === "cut")
    .map((e) => ({
      key: `${e.rung}-${e.name}`,
      rung: e.rung,
      name: e.name,
      title: asText(e.title),
      status: e.status,
      kind: e.status === "cut" ? "cut" : "not adopted",
      verdict: verdictText(e),
    }));

  const g5 = g5Doc?.payload;
  const series = g5?.series ?? [];
  const red = series.filter((s) => s.verdict === "RED");
  if (red.length) {
    const worst = red
      .flatMap((s) =>
        (s.window_excess ?? [])
          .filter((w) => w.role === "adversarial" && Number.isFinite(w.excess_pp))
          .map((w) => ({ ...w, detector: s.detector }))
      )
      .sort((a, b) => b.excess_pp - a.excess_pp)[0];
    const who = red.length === series.length ? `RED on every detector (${series.length})` : `RED on ${red.length} of ${series.length} detectors`;
    entries.push({
      key: "g5",
      rung: null,
      name: "confounder_null",
      title: "The confounder null — charter claim K-1",
      status: "RED",
      kind: "claim falsified",
      verdict: worst
        ? `${who}. On a population with no fraud in it at all, the ${worst.detector} detector's alert rate inside the ${worst.confounder} window (days ${worst.start_day}–${worst.end_day}) ran ${worst.excess_pp.toFixed(2)} pp above nominal, against an allowed ${g5.excess_allowed_pp} pp. Every one of those alerts is a false positive by construction.`
        : `${who}.`,
    });
  }
  return entries;
}

export function deriveLocks(doc) {
  const p = doc?.payload;
  const locks = p?.locks ?? [];
  return {
    locks,
    n: locks.length,
    opens: locks.reduce((n, l) => n + (l.open_count ?? 0), 0),
    authoritative: locks.find((l) => l.authoritative) ?? null,
    preRegistered: locks.filter((l) => l.pre_registration).length,
    opened: Boolean(p?.test_split_opened),
  };
}

export function deriveManifest(doc) {
  const artifacts = doc?.payload?.artifacts ?? [];
  return {
    artifacts,
    present: artifacts.filter((a) => a.status === "PRESENT").length,
    total: artifacts.length,
  };
}

export function deriveJourney(doc) {
  const generations = doc?.payload?.generations ?? [];
  const eras = generations.map((g) => asText(g.era)).filter(Boolean);
  return {
    generations,
    external: generations.filter((g) => g.external).length,
    span: eras.length ? `${eras[0]} – ${eras[eras.length - 1]}`.replace(/^(.+) – \1$/, "$1") : null,
  };
}

// Artifact loader — the site's only data source is the committed, versioned
// artifact contract emitted by `rakshak.artifacts.build` (T-0126) and validated
// by `rakshak.artifacts` (src/rakshak/artifacts/__init__.py). No backend, ever.
//
// The contract is fail-visible: a rejected fetch, a failed HTTP response, or a
// schema mismatch rejects with the artifact name and the reason. No path here
// resolves with a substituted or hardcoded value — that is what stops the site
// from ever displaying a number the artifacts do not support. Ported from
// ver1/dashboard/src/artifacts.js (T-0127, #61): same caching/eviction and
// fail-visible shape, re-pointed at the v3 envelope (`artifact` name +
// `schema_version` string + `split` + `provenance` + `payload`, rather than
// v1's flat `schema_version` integer).
import { useEffect, useState } from "react";

// Pinned to rakshak.artifacts.SCHEMA_VERSION (src/rakshak/artifacts/__init__.py).
// Bump this string exactly when that constant bumps — nothing here infers it.
export const SCHEMA_VERSION = "v3.1.0";

export const SPLIT_VALUES = new Set(["TRAIN", "VALIDATION", "TEST", "NULL_RUN"]);

// name -> committed filename. Every name `rakshak.artifacts.ARTIFACT_NAMES` knows.
const FILES = {
  manifest: "manifest.json",
  lock_state: "lock_state.json",
  ladder: "ladder.json",
  g5_confounder_null: "g5_confounder_null.json",
  rung_roster: "rung_roster.json",
  cost_sweep: "cost_sweep.json",
  journey: "journey.json",
};

// Successful loads are shared; failures are evicted so re-entering a section retries.
const cache = new Map();

export function clearArtifactCache() {
  cache.clear();
}

export function loadArtifact(name) {
  const hit = cache.get(name);
  if (hit) return hit;
  const pending = fetchArtifact(name);
  cache.set(name, pending);
  pending.catch(() => cache.delete(name));
  return pending;
}

async function fetchArtifact(name) {
  const file = FILES[name];
  if (!file) throw new Error(`unknown artifact: ${name}`);

  const base = import.meta.env?.BASE_URL ?? "/";
  const res = await fetch(`${base}artifacts/${file}`);
  if (!res.ok) {
    throw new Error(
      `${name}: HTTP ${res.status} fetching ${file} — MISSING artifact, not a substituted value`
    );
  }

  let doc;
  try {
    doc = await res.json();
  } catch (e) {
    throw new Error(`${name}: ${file} is not parseable JSON (${e.message})`);
  }

  validateEnvelope(name, file, doc);
  return doc; // { artifact, schema_version, split, provenance, payload }
}

// Mirrors the checks `rakshak.artifacts.validate` makes at write time (name,
// schema_version, split-or-null) — not a re-implementation of the full Python
// validator (forbidden-key scan, per-row split checks, roster rules), which
// already ran before the file was committed. This is the loader's own job per
// #61: catch a file that drifted or was hand-edited after that point.
function validateEnvelope(name, file, doc) {
  if (!doc || typeof doc !== "object") {
    throw new Error(`${name}: ${file} top level is not an object`);
  }
  if (doc.artifact !== name) {
    throw new Error(`${name}: ${file} declares artifact ${JSON.stringify(doc.artifact)}, expected ${JSON.stringify(name)}`);
  }
  if (doc.schema_version !== SCHEMA_VERSION) {
    throw new Error(
      `${name}: ${file} schema_version ${JSON.stringify(doc.schema_version)} != expected ${JSON.stringify(SCHEMA_VERSION)}`
    );
  }
  if (doc.split !== null && !SPLIT_VALUES.has(doc.split)) {
    throw new Error(`${name}: ${file} split ${JSON.stringify(doc.split)} is not one of ${[...SPLIT_VALUES].join(", ")} or null`);
  }
  if (!doc.payload || typeof doc.payload !== "object") {
    throw new Error(`${name}: ${file} is missing payload`);
  }
}

/**
 * Load on entry, not eagerly: pass `enabled: false` until the section that
 * needs the artifact is reached (see useInView.js).
 */
export function useArtifact(name, { enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: enabled });
  useEffect(() => {
    if (!enabled) return undefined;
    let alive = true;
    setState((s) => (s.loading ? s : { ...s, loading: true }));
    loadArtifact(name)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }));
    return () => {
      alive = false;
    };
  }, [name, enabled]);
  return state;
}

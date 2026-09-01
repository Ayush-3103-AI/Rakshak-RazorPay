// The loader is the site's data seam, so it is tested at that seam against a
// stubbed fetch — no DOM, no component tree. The load-bearing assertion across
// this group is that no failure path ever resolves: the site must not be able
// to display a number the artifacts do not support. Ported from
// ver1/dashboard/src/artifacts.test.js (T-0127, #61), re-pointed at the v3
// envelope (`artifact` name + string `schema_version` + `split`).
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SCHEMA_VERSION, clearArtifactCache, loadArtifact } from "./artifacts.js";

const ok = (body) => ({
  ok: true,
  status: 200,
  json: async () => body,
});

const envelope = (name, over = {}) => ({
  artifact: name,
  schema_version: SCHEMA_VERSION,
  split: null,
  provenance: {},
  payload: { hello: "world" },
  ...over,
});

const stub = (impl) => vi.stubGlobal("fetch", vi.fn(impl));

beforeEach(() => clearArtifactCache());
afterEach(() => vi.unstubAllGlobals());

describe("loadArtifact", () => {
  it("resolves a well-formed envelope at the expected schema version", async () => {
    stub(async () => ok(envelope("lock_state", { payload: { n_locks: 2 } })));
    await expect(loadArtifact("lock_state")).resolves.toMatchObject({
      payload: { n_locks: 2 },
    });
  });

  it("rejects a schema-version mismatch, naming the artifact and both versions", async () => {
    stub(async () => ok(envelope("ladder", { schema_version: "v2.9.9" })));
    await expect(loadArtifact("ladder")).rejects.toThrow(
      `ladder: ladder.json schema_version "v2.9.9" != expected "${SCHEMA_VERSION}"`
    );
  });

  it("rejects an artifact-name mismatch", async () => {
    stub(async () => ok(envelope("ladder", { artifact: "rung_roster" })));
    await expect(loadArtifact("ladder")).rejects.toThrow(/declares artifact "rung_roster"/);
  });

  it("rejects a split outside the contract's vocabulary", async () => {
    stub(async () => ok(envelope("g5_confounder_null", { split: "holdout" })));
    await expect(loadArtifact("g5_confounder_null")).rejects.toThrow(/split "holdout"/);
  });

  it("accepts every split the contract allows, including NULL_RUN", async () => {
    stub(async () => ok(envelope("g5_confounder_null", { split: "NULL_RUN" })));
    await expect(loadArtifact("g5_confounder_null")).resolves.toMatchObject({
      split: "NULL_RUN",
    });
  });

  it("rejects a non-successful HTTP response, naming the artifact, file and status", async () => {
    stub(async () => ({ ok: false, status: 404 }));
    await expect(loadArtifact("rung_roster")).rejects.toThrow(
      "rung_roster: HTTP 404 fetching rung_roster.json"
    );
  });

  it("rejects rather than resolving when the network fails", async () => {
    stub(async () => {
      throw new TypeError("Failed to fetch");
    });
    await expect(loadArtifact("manifest")).rejects.toThrow("Failed to fetch");
  });

  it("rejects an unknown artifact name", async () => {
    stub(async () => ok(envelope("ladder")));
    await expect(loadArtifact("does_not_exist")).rejects.toThrow("unknown artifact: does_not_exist");
  });

  it("rejects a body that is not the JSON it claims to be", async () => {
    stub(async () => ({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError("Unexpected token");
      },
    }));
    await expect(loadArtifact("ladder")).rejects.toThrow(/not parseable JSON/);
  });

  it("retries after a failure rather than caching it", async () => {
    let attempt = 0;
    stub(async () => {
      attempt += 1;
      return attempt === 1 ? { ok: false, status: 500 } : ok(envelope("manifest"));
    });
    await expect(loadArtifact("manifest")).rejects.toThrow("HTTP 500");
    await expect(loadArtifact("manifest")).resolves.toBeTruthy();
    expect(attempt).toBe(2);
  });

  it("fetches a successfully loaded artifact only once", async () => {
    stub(async () => ok(envelope("manifest")));
    await loadArtifact("manifest");
    await loadArtifact("manifest");
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});

// Number and label formatting shared across sections. Nothing here holds a
// data value — only formatting rules — so it carries no artifact-shaped
// literals of its own (the shell's lint-level discipline from #61's AC).

export function fmtNum(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-IN", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

export function fmtPct(v, digits = 1) {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtSigned(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  const s = v.toLocaleString("en-IN", { maximumFractionDigits: digits });
  return v > 0 ? `+${s}` : s;
}

export function shortSha(sha) {
  if (!sha) return "—";
  return `${sha.slice(0, 10)}…`;
}

// "Infinity" on ttd_median_days means "never detected" — the finding non_finite
// exists to preserve, per build.py's _median(). NaN means "nothing to divide
// by" (e.g. precision with zero alerts). Read the census, don't guess.
export function nonFiniteLabel(census) {
  if (!census) return null;
  const parts = [];
  if (census.Infinity) parts.push(`never (+Inf on ${census.Infinity} seed${census.Infinity === 1 ? "" : "s"})`);
  if (census["-Infinity"]) parts.push(`−Inf on ${census["-Infinity"]} seed${census["-Infinity"] === 1 ? "" : "s"}`);
  if (census.NaN) parts.push(`undefined (NaN on ${census.NaN} seed${census.NaN === 1 ? "" : "s"})`);
  return parts.join(", ") || null;
}

export function splitLabel(split) {
  return split ?? "—";
}

/**
 * Free text out of a YAML-sourced artefact is not guaranteed to be a string.
 * `configs/rung_roster.yaml` has an unquoted citation containing ": ", which
 * YAML parses as a single-key MAP — so `rung_roster.json` really does carry
 * `{"docs/LOGBOOK.md, T-0119 (\"Next": "T-0120 (Rung 5 MIL)\")"}` where a
 * string was meant. React throws on an object child, which took the whole
 * panel down. Rendering it back as `key: value` shows the reader exactly what
 * the file says rather than hiding the defect, and it is one function at the
 * one boundary every roster string passes through.
 */
export function asText(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(asText).join(" ");
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([k, v]) => `${k}: ${asText(v)}`)
      .join(" ");
  }
  return String(value);
}

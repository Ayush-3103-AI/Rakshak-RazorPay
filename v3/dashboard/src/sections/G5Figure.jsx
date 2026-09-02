// #62 — the lead figure. Raw vs cohort-residual alert rate on a freshly
// generated zero-prevalence population, where every alert is a false positive
// by construction. The section states the K-1 reading in prose, sourced from
// each series' own `verdict` field, not from anything this component decides.
//
// docs/10-eval-harness-spec.md §245: "If the lines are the same, charter K-1
// has fired. Publish the figure anyway." So the honest reading is a function
// of the two verdicts, and all four combinations get language of their own —
// including the one where the figure undercuts the project's own hypothesis.
import { motion, useReducedMotion } from "framer-motion";
import { Info } from "lucide-react";
import G5Chart from "../components/G5Chart.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum, fmtPct } from "../lib/format.js";
import { cn } from "../lib/cn.js";

function reading(raw, residual) {
  const rawV = raw?.verdict;
  const resV = residual?.verdict;
  if (!rawV || !resV) {
    return {
      tone: "neutral",
      headline: "One of the two detectors did not report a verdict.",
      body: "The figure below shows whichever series the gate did dump. A verdict is a field on the artefact; this section does not infer one.",
    };
  }
  if (rawV === "RED" && resV === "GREEN") {
    return {
      tone: "good",
      headline: "K-1 has not fired: the cohort-residual detector holds where the raw detector does not.",
      body: "The raw detector alerts above its allowed excess inside the adversarial confounder windows; the cohort-residual detector does not. That difference between the two lines is the whole of charter hypothesis K-1, and on this run it survives.",
    };
  }
  if (rawV === "RED" && resV === "RED") {
    return {
      tone: "bad",
      headline: "K-1 has fired: both detectors alert inside the adversarial windows.",
      body: "Cohort-residual features are not suppressing the confounder signal — the two lines behave the same way where they were supposed to diverge. Per docs/10-eval-harness-spec.md §245 the figure ships anyway: a clean falsification is a result, and this one is stated here rather than buried in a caption.",
    };
  }
  if (rawV === "GREEN" && resV === "GREEN") {
    return {
      tone: "neutral",
      headline: "Neither detector exceeds its allowed excess on this run.",
      body: "The contrast K-1 is measured on is not exercised here: with the raw detector already inside the bound there is no confounder-driven excess for the cohort-residual layer to remove. This is not evidence for or against K-1.",
    };
  }
  return {
    tone: "neutral",
    headline: "The cohort-residual detector is RED where the raw detector is GREEN.",
    body: "The residual layer is alerting inside the confounder windows that the raw detector rides out. That is the reverse of the K-1 prediction and is reported as measured.",
  };
}

const TONE = {
  good: "border-positive-border bg-positive-bg",
  bad: "border-negative-border bg-negative-bg",
  neutral: "border-information-border bg-information-bg",
};

// `eyebrow` is a prop rather than a literal because #79 re-homed this figure
// under §3 (Results) — the section number is now the shell's to decide, not
// this component's. Everything else about the figure is unchanged.
export default function G5Figure({ eyebrow = "§3c · G5 confounder null" }) {
  const g5 = useArtifact("g5_confounder_null");
  const reduce = useReducedMotion();
  const payload = g5.data?.payload;
  const series = payload?.series ?? [];
  const raw = series.find((s) => s.detector === "raw");
  const residual = series.find((s) => s.detector === "cohort-residual");
  const verdictReading = reading(raw, residual);
  const windows = payload?.windows ?? [];

  return (
    <div className="border-b border-border px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          {eyebrow}
        </p>
        <h2 className="m-0 max-w-[26ch] font-heading text-3xl font-bold tracking-tight text-foreground">
          Can it tell platform drift from fraud?
        </h2>
        <p className="mt-[var(--spacing-4)] max-w-[68ch] text-base leading-relaxed text-muted-foreground">
          Run at <strong className="text-foreground">prevalence 0</strong>: there is no fraud in this
          population, so every alert on this chart is a false positive by construction. Shaded bands are
          confounder windows — platform-wide behaviour changes a detector must ride out. Adversarial bands
          (hatched) are the ones designed to fool it; control bands (dotted) are the ones that should be
          uneventful either way.
        </p>

        {g5.loading && <div className="mt-[var(--spacing-6)]"><ArtifactLoading label="Loading g5_confounder_null.json…" /></div>}
        {g5.error && <div className="mt-[var(--spacing-6)]"><ArtifactError artifact="g5_confounder_null" error={g5.error} /></div>}

        {payload && (
          <motion.div
            initial={reduce ? false : { opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.5, ease: [0, 0, 0.2, 1] }}
          >
            <div className={cn("mt-[var(--spacing-6)] rounded-[var(--radius-md)] border-l-4 border p-[var(--spacing-5)]", TONE[verdictReading.tone])}>
              <p className="m-0 text-base font-semibold text-foreground">{verdictReading.headline}</p>
              <p className="m-0 mt-[var(--spacing-3)] max-w-[76ch] text-sm leading-relaxed text-muted-foreground">
                {verdictReading.body}
              </p>
            </div>

            <Card pad="regular" elevation="mid" className="mt-[var(--spacing-6)]">
              <div className="flex flex-wrap items-center gap-[var(--spacing-4)]">
                <h3 className="m-0 font-heading text-lg font-semibold text-foreground">
                  Alert rate by simulation day
                </h3>
                {/* The split label lives on the figure itself, never in a footnote. */}
                <SplitChip split={g5.data.split ?? series[0]?.split} />
                <span className="font-mono text-2xs text-faint">
                  prevalence {fmtNum(payload.prevalence, 3)} · {payload.n_days} days ·{" "}
                  {payload.window_convention}
                </span>
              </div>

              <div className="mt-[var(--spacing-4)] flex flex-wrap gap-[var(--spacing-5)]">
                {series.map((s) => (
                  <div key={s.detector} className="flex items-center gap-[var(--spacing-3)]">
                    <span
                      className="inline-block h-[3px] w-[22px] shrink-0 rounded-full"
                      style={{
                        background: s.detector === "raw" ? "var(--color-negative)" : "var(--color-information)",
                        ...(s.detector === "raw"
                          ? {}
                          : {
                              background:
                                "repeating-linear-gradient(90deg, var(--color-information) 0 7px, transparent 7px 11px)",
                            }),
                      }}
                    />
                    <span className="font-mono text-xs text-foreground">{s.detector}</span>
                    <StatusChip status={s.verdict} />
                    <span className="font-mono text-2xs text-faint">
                      threshold {fmtNum(s.threshold, 2)} · quiet-day {fmtPct(s.quiet_day_rate, 2)}
                    </span>
                  </div>
                ))}
                <div className="flex items-center gap-[var(--spacing-4)] text-2xs text-faint">
                  <span className="flex items-center gap-[var(--spacing-2)]">
                    <span
                      aria-hidden="true"
                      className="inline-block h-[12px] w-[12px] rounded-[2px] border border-negative/40"
                      style={{
                        background:
                          "repeating-linear-gradient(45deg, var(--color-negative-bg) 0 3px, var(--color-negative) 3px 4px)",
                      }}
                    />
                    adversarial window
                  </span>
                  <span className="flex items-center gap-[var(--spacing-2)]">
                    <span
                      aria-hidden="true"
                      className="inline-block h-[12px] w-[12px] rounded-[2px] border border-information/40"
                      style={{ background: "var(--color-information-bg)" }}
                    />
                    control window
                  </span>
                </div>
              </div>

              <div className="mt-[var(--spacing-5)] overflow-x-auto">
                <G5Chart
                  nDays={payload.n_days}
                  windows={windows}
                  series={series}
                  nominalAlertRate={payload.nominal_alert_rate}
                  excessAllowedPp={payload.excess_allowed_pp}
                />
              </div>

              <p className="mt-[var(--spacing-4)] mb-0 flex items-start gap-[var(--spacing-2)] text-xs text-faint">
                <Info className="mt-[1px] h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                Days with no value in <code className="font-mono">alert_rate_by_day</code> are days the
                detector had no baseline yet. They render as a break in the line — never as a zero, which
                would read as "no alerts" rather than "not measurable".
              </p>
            </Card>

            <Card pad="regular" elevation="low" className="mt-[var(--spacing-5)]">
              <h3 className="m-0 font-heading text-lg font-semibold text-foreground">
                Windows, and what each detector did inside them
              </h3>
              <p className="mt-[var(--spacing-2)] mb-0 text-xs text-faint">
                Ranges are half-open, exactly as the artefact declares:{" "}
                <code className="font-mono">{payload.window_convention}</code>. Excess is measured in
                percentage points against the nominal rate, with{" "}
                <strong className="text-foreground">{fmtNum(payload.excess_allowed_pp, 1)}pp</strong> allowed.
              </p>
              <div className="mt-[var(--spacing-4)] overflow-x-auto rounded-[var(--radius-sm)] border border-border">
                <table className="w-full min-w-[720px] border-collapse text-sm">
                  <thead>
                    <tr>
                      {["window", "role", "days", "feature", ...series.map((s) => `${s.detector} excess`)].map((h) => (
                        <th
                          key={h}
                          className="sticky top-0 border-b border-border bg-card px-[var(--spacing-4)] py-[var(--spacing-3)] text-left text-2xs font-semibold tracking-wide text-faint uppercase"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {windows.map((w, i) => (
                      <tr key={`${w.confounder}-${w.start_day}-${i}`} className="border-b border-border last:border-0">
                        <td className="px-[var(--spacing-4)] py-[var(--spacing-3)] font-mono text-xs font-semibold text-foreground">
                          {w.confounder}
                        </td>
                        <td className="px-[var(--spacing-4)] py-[var(--spacing-3)]">
                          <span
                            className={cn(
                              "inline-flex items-center gap-[var(--spacing-1)] rounded-full border px-[var(--spacing-3)] py-[1px] font-mono text-2xs font-bold uppercase",
                              w.role === "adversarial"
                                ? "border-negative-border bg-negative-bg text-negative"
                                : "border-information-border bg-information-bg text-information"
                            )}
                          >
                            {w.role}
                          </span>
                        </td>
                        <td className="tabular px-[var(--spacing-4)] py-[var(--spacing-3)] font-mono text-xs text-muted-foreground">
                          [{w.start_day}, {w.end_day}) · {w.end_day - w.start_day}d
                        </td>
                        <td className="px-[var(--spacing-4)] py-[var(--spacing-3)] font-mono text-xs text-muted-foreground">
                          {w.feature ?? "—"}
                        </td>
                        {series.map((s) => {
                          const hit = (s.window_excess ?? []).find(
                            (e) => e.confounder === w.confounder && e.start_day === w.start_day
                          );
                          const over = hit && payload.excess_allowed_pp != null && hit.excess_pp > payload.excess_allowed_pp;
                          return (
                            <td
                              key={s.detector}
                              className={cn(
                                "tabular px-[var(--spacing-4)] py-[var(--spacing-3)] font-mono text-xs",
                                over ? "font-semibold text-negative" : "text-muted-foreground"
                              )}
                            >
                              {hit ? (
                                <>
                                  {hit.excess_pp > 0 ? "+" : ""}
                                  {fmtNum(hit.excess_pp, 2)}pp
                                  <span className="ml-[var(--spacing-2)] text-faint">
                                    ({fmtPct(hit.alert_rate, 2)})
                                  </span>
                                  {over && <span className="ml-[var(--spacing-2)] not-italic">over</span>}
                                </>
                              ) : (
                                <span className="text-faint italic">not measured</span>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
}

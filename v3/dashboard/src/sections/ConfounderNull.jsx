// §4 — the confounder null (G5), previously G5Figure.jsx under §3c.
//
// The gate runs on a freshly generated zero-prevalence population, where every
// alert is a false positive by construction. The honest reading is a function of
// the two detectors' own `verdict` fields, not of anything this component
// decides, and all four combinations get language of their own — including the
// one where the figure undercuts the project's own hypothesis.
//
// docs/10-eval-harness-spec.md §245: "If the lines are the same, charter K-1 has
// fired. Publish the figure anyway."
import { Info } from "lucide-react";
import G5Chart from "../components/G5Chart.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import SplitChip from "../components/ui/SplitChip.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";
import { fmtNum } from "../lib/format.js";
import { cn } from "../lib/cn.js";

function reading(raw, residual) {
  const rawV = raw?.verdict;
  const resV = residual?.verdict;
  if (!rawV || !resV) {
    return {
      tone: "neutral",
      headline: "One of the two detectors did not report a verdict.",
      body: "The figure shows whichever series the gate did dump. A verdict is a field on the artefact; this section does not infer one.",
    };
  }
  if (rawV === "RED" && resV === "GREEN") {
    return {
      tone: "good",
      headline: "K-1 holds: the cohort-residual detector stays quiet where the raw detector does not.",
      body: "The raw detector alerts above its allowed excess inside the adversarial windows; the residual detector does not. That gap between the two lines is the whole of charter hypothesis K-1, and on this run it survives.",
    };
  }
  if (rawV === "RED" && resV === "RED") {
    return {
      tone: "bad",
      headline: "K-1 has fired: both detectors alert inside the adversarial windows.",
      body: "Cohort-residual features are not suppressing the confounder signal — the two lines behave the same way where they were supposed to diverge. Per spec §245 the figure ships anyway: a clean falsification is a result, stated here rather than buried in a caption.",
    };
  }
  if (rawV === "GREEN" && resV === "GREEN") {
    return {
      tone: "neutral",
      headline: "Neither detector exceeds its allowed excess on this run.",
      body: "The contrast K-1 is measured on is not exercised here: with the raw detector already inside the bound there is no confounder-driven excess for the residual layer to remove. Not evidence for or against K-1.",
    };
  }
  return {
    tone: "neutral",
    headline: "The residual detector is RED where the raw detector is GREEN.",
    body: "The residual layer is alerting inside the confounder windows the raw detector rides out. That is the reverse of the K-1 prediction and is reported as measured.",
  };
}

const TONE = {
  good: "border-positive-border bg-positive-bg",
  bad: "border-negative-border bg-negative-bg",
  neutral: "border-information-border bg-information-bg",
};

export default function ConfounderNull() {
  const g5 = useArtifact("g5_confounder_null");
  const payload = g5.data?.payload;
  const series = payload?.series ?? [];
  const raw = series.find((s) => s.detector === "raw");
  const residual = series.find((s) => s.detector === "cohort-residual");
  const verdictReading = reading(raw, residual);
  const windows = payload?.windows ?? [];

  return (
    <Page
      id="null"
      eyebrow="§4 · Evidence"
      title="Quiet when the whole platform moves."
      lede={
        <>
          Run at <strong className="text-foreground">prevalence 0</strong>: there is no fraud in this
          population, so every alert on this chart is a false positive by construction. Shaded bands are
          platform-wide events — a festival, an outage — that a detector has to ride out.
        </>
      }
      actions={<SplitChip split={g5.data?.split ?? series[0]?.split} />}
    >
      {g5.loading && <ArtifactLoading label="Loading g5_confounder_null.json…" />}
      {g5.error && <ArtifactError artifact="g5_confounder_null" error={g5.error} />}

      {payload && (
        <div className="flex h-full min-h-0 flex-col gap-[var(--spacing-4)]">
          <div
            className={cn(
              "shrink-0 rounded-[var(--radius-md)] border border-l-4 px-[var(--spacing-5)] py-[var(--spacing-4)]",
              TONE[verdictReading.tone]
            )}
          >
            <p className="m-0 text-sm font-semibold text-foreground">{verdictReading.headline}</p>
            <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-muted-foreground">
              {verdictReading.body}
            </p>
          </div>

          <div className="grid min-h-0 flex-1 grid-cols-[1.6fr_1fr] gap-[var(--spacing-4)] max-xl:grid-cols-1">
            <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
              <div className="flex shrink-0 flex-wrap items-center gap-[var(--spacing-4)]">
                <h3 className="m-0 font-heading text-base font-semibold text-foreground">
                  Alert rate by simulation day
                </h3>
                <span className="font-mono text-2xs text-faint">
                  prevalence {fmtNum(payload.prevalence, 3)} · {payload.n_days} days
                </span>
              </div>

              <div className="mt-[var(--spacing-3)] flex shrink-0 flex-wrap gap-[var(--spacing-4)]">
                {series.map((s) => (
                  <div key={s.detector} className="flex items-center gap-[var(--spacing-2)]">
                    <span
                      className="inline-block h-[3px] w-[20px] shrink-0 rounded-full"
                      style={
                        s.detector === "raw"
                          ? { background: "var(--color-negative)" }
                          : {
                              background:
                                "repeating-linear-gradient(90deg, var(--color-information) 0 7px, transparent 7px 11px)",
                            }
                      }
                    />
                    <span className="font-mono text-2xs text-foreground">{s.detector}</span>
                    <StatusChip status={s.verdict} />
                  </div>
                ))}
              </div>

              <div className="mt-[var(--spacing-4)] min-h-0 flex-1 overflow-auto">
                <G5Chart
                  nDays={payload.n_days}
                  windows={windows}
                  series={series}
                  nominalAlertRate={payload.nominal_alert_rate}
                  excessAllowedPp={payload.excess_allowed_pp}
                />
              </div>

              <p className="m-0 mt-[var(--spacing-3)] flex shrink-0 items-start gap-[var(--spacing-2)] text-2xs text-faint">
                <Info className="mt-[1px] h-3 w-3 shrink-0" aria-hidden="true" />
                Days with no baseline yet render as a break in the line — never as a zero, which would
                read as "no alerts" rather than "not measurable".
              </p>
            </Card>

            <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
              <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">
                What each detector did inside the windows
              </h3>
              <p className="m-0 mt-[var(--spacing-2)] shrink-0 text-2xs text-faint">
                Excess in percentage points against nominal, with{" "}
                <strong className="text-foreground">{fmtNum(payload.excess_allowed_pp, 1)}pp</strong>{" "}
                allowed. Ranges half-open: <code className="font-mono">{payload.window_convention}</code>.
              </p>

              <div className="mt-[var(--spacing-4)] min-h-0 flex-1 overflow-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr>
                      {["window", "role", ...series.map((s) => s.detector)].map((h) => (
                        <th
                          key={h}
                          className="sticky top-0 border-b border-border bg-card py-[var(--spacing-3)] pr-[var(--spacing-3)] text-left font-mono text-2xs font-bold tracking-wide text-faint uppercase"
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {windows.map((w, i) => (
                      <tr
                        key={`${w.confounder}-${w.start_day}-${i}`}
                        className="border-b border-border last:border-0"
                      >
                        <td className="py-[var(--spacing-3)] pr-[var(--spacing-3)] font-mono text-2xs font-semibold text-foreground">
                          {w.confounder}
                          <span className="ml-[var(--spacing-2)] font-normal text-faint">
                            [{w.start_day},{w.end_day})
                          </span>
                        </td>
                        <td className="py-[var(--spacing-3)] pr-[var(--spacing-3)]">
                          <span
                            className={cn(
                              "inline-flex rounded-full border px-[var(--spacing-2)] py-[1px] font-mono text-2xs font-bold uppercase",
                              w.role === "adversarial"
                                ? "border-negative-border bg-negative-bg text-negative"
                                : "border-information-border bg-information-bg text-information"
                            )}
                          >
                            {w.role}
                          </span>
                        </td>
                        {series.map((s) => {
                          const hit = (s.window_excess ?? []).find(
                            (e) => e.confounder === w.confounder && e.start_day === w.start_day
                          );
                          const over =
                            hit && payload.excess_allowed_pp != null && hit.excess_pp > payload.excess_allowed_pp;
                          return (
                            <td
                              key={s.detector}
                              className={cn(
                                "tabular py-[var(--spacing-3)] pr-[var(--spacing-3)] font-mono text-2xs",
                                over ? "font-semibold text-negative" : "text-muted-foreground"
                              )}
                            >
                              {hit ? (
                                <>
                                  {hit.excess_pp > 0 ? "+" : ""}
                                  {fmtNum(hit.excess_pp, 2)}pp
                                  {over && <span className="ml-[var(--spacing-1)]">over</span>}
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
          </div>
        </div>
      )}
    </Page>
  );
}

// Screen 5 — the margin is not a point estimate.
//
// Re-price a wrong HOLD against a missed fraud across the artifact's whole
// ratio grid, refit nothing, and watch the survivor's line stay above the
// floor. The line draws left to right on arrival rather than appearing whole,
// because the point of the figure is that it never dips — and a line you watch
// being drawn makes that claim in a way a finished line does not.
//
// The decision-layer share is derived, not stored: the HOLD decomposition's
// delta over the margin above the floor, both at the ratio the decomposition
// was taken at. If the artifact lacks any part of that, the sentence drops
// its number rather than inventing one.
import { ArtifactError, ArtifactLoading } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { fmtNum } from "../../lib/format.js";
import { deriveLadder, deriveSweep } from "../derive.js";
import SweepFigure from "../figures/SweepFigure.jsx";
import Screen, { Chip, Eyebrow, Glass, Headline, Lede, Reveal } from "../Screen.jsx";

export default function StressTest() {
  const ladder = useArtifact("ladder");
  const sweepDoc = useArtifact("cost_sweep");
  const { survivorLabel } = deriveLadder(ladder.data);
  const sweep = deriveSweep(sweepDoc.data, survivorLabel);
  const spanText = sweep?.span ? `${fmtNum(sweep.span, 0)}×` : "orders of magnitude";

  return (
    <Screen id="stress" play={2.6}>
      {(progress) => (
        <div className="grid grid-cols-[5fr_7fr] items-center gap-[clamp(24px,4vw,64px)] max-lg:grid-cols-1">
          <div>
            <Eyebrow>The stress test · {sweep?.split ?? "—"} split</Eyebrow>
            <Headline>
              Change the price of being wrong by {spanText}. <span className="text-primary-text">It still wins.</span>
            </Headline>
            <Lede>
              {sweep?.series && sweep.band ? (
                <>
                  Across {sweep.ratios.length} cost settings spanning {spanText}, with nothing refitted,{" "}
                  <span className="font-mono font-semibold text-foreground">{sweep.policy}</span>&rsquo;s savings hold
                  between{" "}
                  <span className="font-mono font-semibold text-foreground">
                    {fmtNum(sweep.band[0], 4)} and {fmtNum(sweep.band[1], 4)}
                  </span>
                  {sweep.floor && (
                    <>
                      , over a floor of{" "}
                      <span className="font-mono font-semibold text-foreground">{fmtNum(sweep.floor.values[0], 4)}</span>
                    </>
                  )}
                  {sweep.beatsAt != null && (
                    <>
                      {" "}
                      &mdash; ahead at {sweep.beatsAt} of {sweep.ratios.length}.
                    </>
                  )}
                  {sweep.decisionShare != null && (
                    <>
                      {" "}
                      About <span className="font-mono font-semibold text-foreground">{Math.round(sweep.decisionShare * 100)}%</span> of
                      that margin is the decision layer, the right to HOLD, not the ranker.
                    </>
                  )}
                </>
              ) : (
                "The cost sweep has not been scored in this tree, so no margin band is claimed here."
              )}
            </Lede>
            {sweep?.shippedRatio != null && (
              <Reveal className="mt-[var(--spacing-8)]">
                <Chip tone={sweep.shippedWithin ? "notice" : "negative"}>
                  shipped cost matrix at {fmtNum(sweep.shippedRatio, 3)} · {sweep.shippedWithin ? "inside the grid" : "outside the grid"}
                </Chip>
              </Reveal>
            )}
          </div>

          <div className="grid gap-[var(--spacing-4)]">
            {sweepDoc.loading && <ArtifactLoading label="Loading the cost sweep…" />}
            {sweepDoc.error && <ArtifactError artifact="cost_sweep" error={sweepDoc.error} />}
            {sweep?.series && (
              <Glass className="p-[clamp(12px,1.6vw,24px)]">
                <SweepFigure progress={progress} sweep={sweep} />
              </Glass>
            )}
          </div>
        </div>
      )}
    </Screen>
  );
}

// Screen 2 — Razorpay's two guards, and the hole between them.
//
// This is the screen that shows the product was understood, so it is the one
// with the most deliberate picture. It plays on arrival: the axis draws,
// Bumblebee stamps day zero, Vulcan's comb sweeps under the line, the empty
// stretch gets its name, the chargeback lands, and only then does Rakshak's
// row of daily decisions appear over the gap.
import GapAxis from "../figures/GapAxis.jsx";
import Screen, { Eyebrow, Glass, Headline, Lede, Reveal } from "../Screen.jsx";

const GUARDS = [
  { name: "Vulcan", tone: "text-information", what: "Scores every transaction, in milliseconds." },
  { name: "Bumblebee", tone: "text-positive", what: "Reviews every merchant once, at onboarding." },
  { name: "Rakshak", tone: "text-primary-text", what: "Watches every cleared merchant, every day after." },
];

export default function ProductHole() {
  return (
    <Screen id="product" play={2.8}>
      {(progress) => (
        <div className="grid grid-cols-[5fr_7fr] items-center gap-[clamp(24px,4vw,64px)] max-lg:grid-cols-1">
          <div>
            <Eyebrow>The product · where Rakshak sits</Eyebrow>
            <Headline>
              Razorpay has two guards. <span className="text-primary-text">The hole is between them.</span>
            </Headline>
            <Lede>
              Nothing watches a merchant that already cleared onboarding drift over the weeks that follow. Bust-outs,
              laundering endpoints and refund collusion begin after approval and are first seen when the chargebacks
              arrive.
            </Lede>
            <Reveal as="ul" className="m-0 mt-[var(--spacing-8)] grid list-none gap-[var(--spacing-4)] p-0 max-lg:grid-cols-3 max-sm:grid-cols-1">
              {GUARDS.map((g) => (
                <li key={g.name} className="flex items-baseline gap-[var(--spacing-4)]">
                  <span className={`w-[11ch] shrink-0 font-mono text-[11px] font-bold tracking-[0.16em] uppercase ${g.tone}`}>
                    {g.name}
                  </span>
                  <span className="text-[length:var(--text-body)] leading-snug text-muted-foreground">{g.what}</span>
                </li>
              ))}
            </Reveal>
          </div>
          <Glass className="p-[clamp(12px,1.6vw,24px)]">
            <GapAxis progress={progress} />
          </Glass>
        </div>
      )}
    </Screen>
  );
}

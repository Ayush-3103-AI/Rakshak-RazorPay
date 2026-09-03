// Screen 1 — the problem, in one breath.
//
// Five seconds: one sentence, one line under it, and the scroll cue. The
// result is on screen four, not here, and that is a decision — a reader who
// opens on a number has no idea what it is a number OF. But a problem screen
// that becomes a preamble loses the same reader, so nothing on this screen is
// allowed to be a paragraph.
//
// The lock chip is the one artifact read on this screen, and the one claim a
// front door can make before showing any evidence: the harness was sealed and
// the test split has been opened this many times.
import { motion, useReducedMotion } from "framer-motion";
import { ChevronDown, Lock } from "lucide-react";
import { useArtifact } from "../../lib/artifacts.js";
import { deriveLocks } from "../derive.js";
import Screen, { Chip, Eyebrow, Headline, Lede, Reveal } from "../Screen.jsx";

export default function Gap() {
  const reduce = useReducedMotion();
  const lockState = useArtifact("lock_state");
  const locks = deriveLocks(lockState.data);

  return (
    <Screen id="gap" contentClassName="pt-[4vh]">
      <Eyebrow>Razorpay AI Buildathon 2026 · Track 02 · AI Risk Manager</Eyebrow>
      <Headline as="h1" size="xl">
        A merchant clears onboarding. <span className="text-primary-text">Then nobody watches.</span>
      </Headline>
      <Lede>
        Fraud that begins <em>after</em> approval surfaces as chargebacks 45 to 120 days later. Rakshak is
        the sentinel for that gap: one decision per merchant, per day, with a reason attached.
      </Lede>

      <Reveal className="mt-[var(--spacing-9)] flex flex-wrap gap-[var(--spacing-3)]">
        <Chip>Solo build</Chip>
        <Chip>Synthetic data · no Razorpay data or APIs</Chip>
        {lockState.data && (
          <Chip tone="primary">
            <Lock aria-hidden="true" className="h-3 w-3" />
            {locks.n} sealed locks · test split opened {locks.opens}×
          </Chip>
        )}
      </Reveal>

      <Reveal className="mt-[clamp(32px,8vh,80px)] flex items-center gap-[var(--spacing-4)] font-mono text-xs tracking-[0.14em] text-faint uppercase">
        <motion.span
          aria-hidden="true"
          className="grid h-9 w-9 place-items-center rounded-full border border-border"
          animate={reduce ? undefined : { y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: "easeInOut" }}
        >
          <ChevronDown className="h-4 w-4" />
        </motion.span>
        Scroll · a two-minute read
      </Reveal>
    </Screen>
  );
}

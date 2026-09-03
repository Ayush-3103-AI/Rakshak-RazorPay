// Screen 4 — the result, as a sentence the artifacts complete.
//
// "{N} policies raced. One survived every test." The count comes from the
// ladder's rows, the survivor from `beats_all_floors`, the seed count from the
// thinnest row, the lock count from the lock chain. Promote a second row
// upstream and this screen says "Two survived" with nobody editing copy —
// which is the point of not typing numbers into a headline.
import { ArtifactError, ArtifactLoading } from "../../components/ui/ArtifactState.jsx";
import { useArtifact } from "../../lib/artifacts.js";
import { deriveLadder, deriveLocks } from "../derive.js";
import MarginBars from "../figures/MarginBars.jsx";
import Screen, { Chip, Eyebrow, Glass, Headline, Lede, Reveal, Stat } from "../Screen.jsx";

const WORDS = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];

export default function Result() {
  const ladder = useArtifact("ladder");
  const lockState = useArtifact("lock_state");
  const d = deriveLadder(ladder.data);
  const locks = deriveLocks(lockState.data);
  const n = d.rungs.length;
  const s = d.survivors.length;

  return (
    <Screen id="result">
      <Eyebrow>
        The result · {d.split ?? "—"} split · {d.seedCount} seeds
      </Eyebrow>
      <Headline>
        {n ? (
          <>
            {n} policies raced.{" "}
            <span className="text-primary-text">{WORDS[s] ?? s} survived every test.</span>
          </>
        ) : (
          "The ladder has not been scored in this tree."
        )}
      </Headline>
      <Lede>
        Each policy was scored against {d.floors.length || "the"} named floors, including one that needs no model at
        all: rank merchants by size and alert on the biggest.{" "}
        {d.survivorLabel ? (
          <>
            <span className="font-mono font-semibold text-foreground">{d.survivorLabel}</span> is the only row that beat
            every floor on every one of its {d.survivors[0]?.n_seeds ?? d.seedCount} seeds.
          </>
        ) : (
          "No row beat every floor."
        )}
      </Lede>

      {ladder.loading && <ArtifactLoading label="Loading the ladder…" />}
      {ladder.error && <ArtifactError artifact="ladder" error={ladder.error} />}

      <div className="mt-[var(--spacing-10)] grid grid-cols-[4fr_8fr] gap-[clamp(20px,3vw,48px)] max-lg:grid-cols-1">
        <Reveal className="grid grid-cols-2 gap-x-[var(--spacing-6)] gap-y-[var(--spacing-9)] max-lg:grid-cols-4 max-sm:grid-cols-2">
          <Stat value={s} label={s === 1 ? "row beat every floor" : "rows beat every floor"} note={d.survivorLabel ?? "none"} size="lg" accent className="col-span-2 max-lg:col-span-4 max-sm:col-span-2" />
          <Stat value={n} label="policies scored" note="against the same floors" size="sm" />
          <Stat value={d.seedCount} label={d.seedsUniform ? "seeds per policy" : "seeds, thinnest row"} note="scored on each, not averaged over one" size="sm" />
          <Stat value={locks.n} label="sealed eval locks" note={locks.authoritative?.file ?? "—"} size="sm" />
          <Stat value={locks.opens} label="test split opened" note="the number that keeps every figure honest" size="sm" />
        </Reveal>

        <Glass>
          <div className="flex flex-wrap items-baseline justify-between gap-[var(--spacing-3)]">
            <p className="m-0 font-mono text-[11px] font-bold tracking-[0.18em] text-faint uppercase">
              Savings per policy, against its floor
            </p>
            <Chip tone="primary">
              {s} of {n} beat every floor on every seed
            </Chip>
          </div>
          <div className="mt-[var(--spacing-5)]">
            <MarginBars rows={d.rows} />
          </div>
          <p className="m-0 mt-[var(--spacing-4)] font-mono text-[10px] leading-relaxed tracking-[0.08em] text-faint uppercase">
            <span className="text-primary-text">blue</span> beats every floor on every seed ·{" "}
            <span className="text-notice">amber</span> ahead on the mean, behind on at least one seed ·{" "}
            grey behind the floor
          </p>
        </Glass>
      </div>
    </Screen>
  );
}

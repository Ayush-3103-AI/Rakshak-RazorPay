// §2 — how the numbers are guarded, argued BEFORE any number is shown (#79).
//
// This section exists because the previous shell had nowhere to put it. The lock
// panel was buried two thirds of the way down a section titled "Trajectory &
// ladder", where a reader met the ladder's figures first and the reason to trust
// them second, if at all. The reordering is the point: a risk operator who has
// seen a thousand backtests needs the harness before the result.
//
// Nothing here is asserted in prose that the artifacts do not carry. The lock
// chain, its hashes, the open counter and each cycle's pre-registration document
// come from lock_state.json; the floor vocabulary is derived from the ladder's
// own metric keys, so a floor added upstream appears here without an edit.
import { motion, useReducedMotion } from "framer-motion";
import { FileCheck2, KeyRound, Ruler } from "lucide-react";
import LockPanel from "../components/LockPanel.jsx";
import Card from "../components/ui/Card.jsx";
import StatusChip from "../components/ui/StatusChip.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";

// What each floor asks, in the reader's language. The LIST is derived from the
// ladder; this map only supplies the gloss, and an unglossed floor still renders.
const FLOOR_GLOSS = {
  all_pass: "Alert on nobody. The do-nothing baseline every savings figure is measured against.",
  all_hold: "Hold everyone. The other trivial extreme, and the one that bankrupts the platform.",
  random: "Alert on K merchants at random. Beats a fitted model more often than anyone expects.",
  random_at_k: "Alert on K merchants at random. Beats a fitted model more often than anyone expects.",
  volume_rank:
    "Rank merchants by transaction volume. No learning at all — and the hardest floor on this ladder.",
};

function prettyFloor(key) {
  return key.replace(/^savings_floor_/, "");
}

export default function Method() {
  const lockState = useArtifact("lock_state");
  const ladder = useArtifact("ladder");
  const reduce = useReducedMotion();

  const payload = lockState.data?.payload;
  const locks = payload?.locks ?? [];
  const preRegistered = locks.filter((l) => l.pre_registration);
  const metricKeys = ladder.data?.payload?.metric_keys ?? [];
  const floors = metricKeys.filter((k) => k.startsWith("savings_floor_")).map(prettyFloor);

  const PILLARS = [
    {
      Icon: KeyRound,
      title: "The harness is sealed before the models exist",
      body: (
        <>
          Each cycle hashes the eval module, the generator module and the scenario config into a lock
          file, records the commit that froze it, and carries an <em>open counter</em> for the test
          split. The chain below supersedes forward and is checkable rather than asserted — and the
          counter reads{" "}
          <strong className="text-foreground">
            {payload?.test_split_opened ? "opened" : "0, across every lock"}
          </strong>
          .
        </>
      ),
    },
    {
      Icon: FileCheck2,
      title: "Claims are pre-registered, then reported either way",
      body: (
        <>
          {preRegistered.length} of {locks.length} locks name a pre-registration document written{" "}
          <em>before</em> the run that tested them. When a gate failed, it was reported as a failure —
          including a cycle-4 gate that turned out to be anchored to a threshold its own regeneration
          had invalidated, recorded as a pre-registration error rather than quietly re-anchored.
        </>
      ),
    },
    {
      Icon: Ruler,
      title: "Every policy is scored against explicit floors",
      body: (
        <>
          A model that beats nothing is not a result. Every row on the ladder is priced against{" "}
          {floors.length || "the"} named floors, and a row losing to one is marked as losing{" "}
          <em>to that named floor</em> rather than quietly ranked above it.
        </>
      ),
    },
  ];

  return (
    <div className="border-b border-border bg-canvas-well px-[var(--spacing-8)] py-[var(--spacing-10)] max-md:px-[var(--spacing-5)] max-md:py-[var(--spacing-7)]">
      <div className="mx-auto max-w-[1180px]">
        <p className="m-0 mb-[var(--spacing-3)] font-mono text-xs font-bold tracking-[0.16em] text-primary-text uppercase">
          §2 · How it is measured
        </p>
        <h2 className="m-0 max-w-[30ch] font-heading text-3xl font-bold tracking-tight text-foreground">
          Read this before you read a single number
        </h2>
        <p className="mt-[var(--spacing-4)] max-w-[72ch] text-base leading-relaxed text-muted-foreground">
          Any project can show you a chart where its model wins. The only thing that separates this
          one from that is what was fixed <em>before</em> the chart existed — so the harness comes
          first here, and the results come after it.
        </p>

        <div className="mt-[var(--spacing-8)] grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-[var(--spacing-4)]">
          {PILLARS.map((p, i) => (
            <motion.div
              key={p.title}
              initial={reduce ? false : { opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ delay: reduce ? 0 : i * 0.08, duration: 0.5, ease: [0, 0, 0.2, 1] }}
            >
              <Card pad="regular" elevation="low" className="h-full">
                <p.Icon aria-hidden="true" className="h-5 w-5 text-primary" />
                <h3 className="m-0 mt-[var(--spacing-3)] font-heading text-base font-semibold text-foreground">
                  {p.title}
                </h3>
                <p className="m-0 mt-[var(--spacing-2)] text-sm leading-relaxed text-muted-foreground">{p.body}</p>
              </Card>
            </motion.div>
          ))}
        </div>

        {floors.length > 0 && (
          <div className="mt-[var(--spacing-8)]">
            <h3 className="m-0 mb-[var(--spacing-4)] font-heading text-xl font-bold text-foreground">
              The floors, named
            </h3>
            <div className="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-[var(--spacing-4)]">
              {floors.map((floor) => (
                <Card key={floor} pad="compact" elevation="low">
                  <code className="font-mono text-sm font-bold text-foreground">{floor}</code>
                  <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-muted-foreground">
                    {FLOOR_GLOSS[floor] ?? "Reported on every row; see the ladder for its column."}
                  </p>
                </Card>
              ))}
            </div>
          </div>
        )}

        <div className="mt-[var(--spacing-9)]">
          <div className="mb-[var(--spacing-2)] flex flex-wrap items-center gap-[var(--spacing-3)]">
            <h3 className="m-0 font-heading text-xl font-bold text-foreground">The lock chain</h3>
            <StatusChip status={payload?.test_split_opened ? "MISSING" : "PRESENT"} />
            <span className="font-mono text-2xs text-faint">
              {payload?.test_split_opened
                ? "the test split has been opened"
                : "the test split has never been opened"}
            </span>
          </div>
          <p className="mt-0 mb-[var(--spacing-4)] max-w-[72ch] text-sm text-muted-foreground">
            Three hashes, the open counter, the freezing commit and the pre-registration document, for
            every lock in the chain.
          </p>
          {lockState.loading && <ArtifactLoading label="Loading lock_state.json…" />}
          {lockState.error && <ArtifactError artifact="lock_state" error={lockState.error} />}
          {lockState.data && <LockPanel lockState={lockState.data} variant="full" />}
        </div>
      </div>
    </div>
  );
}

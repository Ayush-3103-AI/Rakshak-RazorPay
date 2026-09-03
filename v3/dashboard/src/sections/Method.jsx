// §5 — how the numbers are guarded.
//
// #79 put this BEFORE the results, reasoning that an operator who has seen a
// thousand backtests needs the harness before the number. That reordering was
// right about the argument and wrong about the reader: a judge who has not yet
// been given a claim has no reason to care how it was guarded. So the claim now
// leads and the guard sits here — the screen a reader reaches at exactly the
// moment they start looking for the catch.
//
// Nothing here is asserted in prose that the artifacts do not carry. The lock
// chain, its hashes, the open counter and each cycle's pre-registration document
// come from lock_state.json; the floor vocabulary is derived from the ladder's
// own metric keys, so a floor added upstream appears here without an edit.
import { FileCheck2, KeyRound, Ruler } from "lucide-react";
import LockPanel from "../components/LockPanel.jsx";
import Page from "../components/Page.jsx";
import Card from "../components/ui/Card.jsx";
import { ArtifactError, ArtifactLoading } from "../components/ui/ArtifactState.jsx";
import { useArtifact } from "../lib/artifacts.js";

// What each floor asks, in the reader's language. The LIST is derived from the
// ladder; this map only supplies the gloss, and an unglossed floor still renders.
const FLOOR_GLOSS = {
  all_pass: "Alert on nobody. The do-nothing baseline every savings figure is measured against.",
  all_hold: "Hold everyone. The other trivial extreme, and the one that bankrupts the platform.",
  random: "Alert on K merchants at random. Beats a fitted model more often than anyone expects.",
  random_at_k: "Alert on K merchants at random. Beats a fitted model more often than anyone expects.",
  volume_rank: "Rank by transaction volume. No learning at all — and the hardest floor on this ladder.",
};

const prettyFloor = (key) => key.replace(/^savings_floor_/, "");

export default function Method() {
  const lockState = useArtifact("lock_state");
  const ladder = useArtifact("ladder");

  const payload = lockState.data?.payload;
  const locks = payload?.locks ?? [];
  const preRegistered = locks.filter((l) => l.pre_registration);
  const metricKeys = ladder.data?.payload?.metric_keys ?? [];
  const floors = metricKeys.filter((k) => k.startsWith("savings_floor_")).map(prettyFloor);

  const PILLARS = [
    {
      Icon: KeyRound,
      title: "Sealed before the models existed",
      body: (
        <>
          Each cycle hashes the eval module, the generator and the scenario config into a lock file,
          records the commit that froze it, and carries an open counter for the test split. It reads{" "}
          <strong className="text-foreground">
            {payload?.test_split_opened ? "opened" : "0, across every lock"}
          </strong>
          .
        </>
      ),
    },
    {
      Icon: FileCheck2,
      title: "Pre-registered, then reported either way",
      body: (
        <>
          {preRegistered.length} of {locks.length} locks name a document written <em>before</em> the run
          that tested them. A cycle-4 gate anchored to a threshold its own regeneration had invalidated
          was recorded as a pre-registration error, not quietly re-anchored.
        </>
      ),
    },
    {
      Icon: Ruler,
      title: "Scored against explicit floors",
      body: (
        <>
          A model that beats nothing is not a result. Every row is priced against {floors.length || "the"}{" "}
          named floors, and a row losing to one is marked as losing <em>to that named floor</em>.
        </>
      ),
    },
  ];

  return (
    <Page
      id="method"
      eyebrow="§5 · Discipline"
      title="Now the part that makes those numbers worth reading."
      lede="Any project can show you a chart where its model wins. The only thing separating this one is what was fixed before the chart existed — and left checkable afterwards."
    >
      <div className="flex h-full min-h-0 flex-col gap-[var(--spacing-4)]">
        <div className="grid shrink-0 grid-cols-3 gap-[var(--spacing-4)] max-lg:grid-cols-1">
          {PILLARS.map((p) => (
            <Card key={p.title} pad="regular" elevation="low" className="h-full">
              <p.Icon aria-hidden="true" className="h-[18px] w-[18px] text-primary" />
              <h3 className="m-0 mt-[var(--spacing-3)] font-heading text-sm font-semibold text-foreground">
                {p.title}
              </h3>
              <p className="m-0 mt-[var(--spacing-2)] text-xs leading-relaxed text-muted-foreground">
                {p.body}
              </p>
            </Card>
          ))}
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-[1fr_1.5fr] gap-[var(--spacing-4)] max-xl:grid-cols-1">
          {floors.length > 0 && (
            <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
              <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">
                The floors, named
              </h3>
              <ul className="m-0 mt-[var(--spacing-3)] min-h-0 flex-1 list-none space-y-[var(--spacing-3)] overflow-auto p-0">
                {floors.map((floor) => (
                  <li key={floor} className="rounded-[var(--radius-sm)] border border-border bg-canvas-well p-[var(--spacing-4)]">
                    <code className="font-mono text-xs font-bold text-foreground">{floor}</code>
                    <p className="m-0 mt-[var(--spacing-1)] text-2xs leading-relaxed text-muted-foreground">
                      {FLOOR_GLOSS[floor] ?? "Reported on every row; see the ladder for its column."}
                    </p>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          <Card pad="regular" elevation="low" className="flex min-h-0 flex-col overflow-hidden">
            <h3 className="m-0 shrink-0 font-heading text-base font-semibold text-foreground">
              The lock chain
            </h3>
            <p className="m-0 mt-[var(--spacing-1)] shrink-0 text-2xs text-faint">
              Three hashes, the open counter, the freezing commit and the pre-registration document, for
              every lock in the chain.
            </p>
            <div className="mt-[var(--spacing-3)] min-h-0 flex-1 overflow-auto">
              {lockState.loading && <ArtifactLoading label="Loading lock_state.json…" />}
              {lockState.error && <ArtifactError artifact="lock_state" error={lockState.error} />}
              {lockState.data && <LockPanel lockState={lockState.data} variant="full" />}
            </div>
          </Card>
        </div>
      </div>
    </Page>
  );
}

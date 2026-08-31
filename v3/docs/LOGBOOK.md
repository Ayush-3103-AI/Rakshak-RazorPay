<!-- HEAD
FILE:     docs/LOGBOOK.md
PHASE:    5 — EXECUTE
STATUS:   append-only. Never edit a past entry.
SUMMARY:  One entry per ticket. The "Surprised me" field is the most valuable thing in
          this repo — it is where the project's model of the world was wrong, and it is
          what the retrospective and the panel round are built from. Record surprises
          even when they are embarrassing. Especially then.
-->

# LOGBOOK — Rakshak v2

Entry template. Copy it, fill it, append. Do not edit entries above.

```
## T-0000 | <title>
Date:        2026-09-0X  ·  Session:  N  ·  Duration:  Xh
Status:      DONE | BLOCKED | PARTIAL (split into T-000Xa/b)

Built:       <what exists now that did not before, by file>
Verified:    <the test that was run, and its output>
Surprised:   <what you expected vs what happened. Blank is a smell — something always
              surprises. If nothing did, you probably did not look.>
Broke:       <what failed, and how it was fixed. Include dead ends.>
Decided:     <any choice made that was not in the spec, and why. If it contradicts a
              spec section, name the section — that is a DESCEND candidate.>
Numbers:     <any measurement taken: timings, sizes, metric values on VALIDATION only>
Next:        <the ticket that follows, named but NOT started>
```

---

## Standing rules

- One entry per ticket, written before the session ends, not batched later.
- Numbers from the **validation** split only until T-151. Any test-split number appearing
  in this file before T-151 is an integrity breach and must be reported, not deleted.
- A BLOCKED entry stops the sprint. Write what is blocking and what would unblock it.
- If a ticket is split, log the split and why — oversized tickets are a planning signal
  worth harvesting.

---

<!-- entries below this line -->

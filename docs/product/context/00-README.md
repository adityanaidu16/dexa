# Ground-truth context directory — how to read it

Compiled 2026-09-03 from two repositories and from web research. Every file here is an INPUT
to the product briefs; nothing here is a product decision.

| file | what it is | how to use it |
|---|---|---|
| 01-evidence-ledger-dexa.md | Every measured claim in /home/user/dexa, one entry per tested thesis, with status PROVEN / BOUNDED / FALSIFIED / OPEN / CONTRADICTED, exact numbers, setup, source file, caveats | Cite numbers from here with the entry id. Respect the status label. Read the provenance paragraph at the top first: it lists the places where the repo's own docs disagree with each other. |
| 02-evidence-ledger-voice.md | Same for /home/user/voice-inference (voice-call KV tiering, arms A-D) | Same. Note the 13 doc-vs-data discrepancies at the top (e.g. arm C's archived N=300 run lost its server at minute 41; full-load-window p95 is ~20-25 ms higher than the whole-run p95 the docs quote). |
| 03-build-inventory.md | What code actually exists in both repos and its real state (tests run where possible) | Use for the 'reuse from repos' part of a build plan. Do not describe scaffold as production. |
| 04-conflicts.md | Contradictions and superseded claims across both repos, with the experiment that resolves each | Never cite a superseded number. When two measurements disagree, say so and cite both. |
| 05-research-*.md | External facts with URLs and as-of dates (Wafer, Morph, generalized providers, frontier caching TTLs, KV-cache ecosystem, voice AI infra, agent infra, doc AI, GPU pricing) | Cite by URL. Confidence levels are the researcher's. These are facts, not judgments. |

Status vocabulary:
- PROVEN — measured and (usually) reproduced; the number can be cited with its setup.
- BOUNDED — measured, but a ceiling or a strong caveat limits what can be claimed.
- FALSIFIED — the thesis as stated was tested and failed.
- OPEN — designed or planned, not measured.
- CONTRADICTED — two measurements in the repos point different ways; cite both.

The founder's rule for everything downstream: no market-size judgments, no "already exists so
don't build it" judgments. Those go into a 'founder decisions' list with the options and the
evidence that would inform them.

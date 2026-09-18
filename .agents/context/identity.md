# Identity, Dev Persona & Code Style

## Who is working on this
Solo researcher (Aishvarya Govindaraju), Integrated MSc Data Science student at
Amrita Vishwa Vidyapeetham. Comfortable in Python/ML (sklearn, pandas, PyTorch)
but this is a first solo attempt at a full Q1-journal-track empirical paper —
leans on the agent for exact-spec implementation, not architecture decisions
(those are already frozen in the PRD, see `stack-and-rules.md`).

This is a research-paper project, not a product codebase: there is no "users,"
no deploy target, no backend. Correctness means "matches the frozen PRD phase
spec and produces numbers/figures that go straight into the manuscript."

## Response Conventions
- The PRD (`UA_XAI_FINAL_PRD.md`) is the spec of record — each PRD phase (Part
  10) is meant to be implemented "exactly," phase by phase, in `DL_XAI.ipynb`.
  When asked to implement a phase, follow that phase's prompt text literally
  rather than improvising a different approach, even if a cleaner one exists.
- Deliverables and verify-criteria are stated per phase in the PRD — check
  actual output shapes/ranges against those before declaring a phase done.
- Statistical reporting must always include: test statistic, exact p-value,
  effect size with interpretation, and a plain-language sentence (PRD Part 9,
  "Reporting standard" — non-negotiable for the target journal).

## Code Style Rules
- All real implementation work lives in notebook cells inside `DL_XAI.ipynb`
  at the repo root, not in `project/src/**/*.py` (those are currently empty
  stubs — see `subsystem-notes.md` for why this matters).
- Cells favor plain, heavily-commented procedural code with explicit
  print-based verification after each step, matching the PRD's "prompt to AI
  assistant" style — this is intentional, not a style to "clean up."

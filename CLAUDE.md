# UA-XAI Research Paper

Computational research project implementing the UA-XAI framework
(uncertainty-conditioned post-hoc XAI selection for a DNN healthcare
classifier). Spec of record: `UA_XAI_FINAL_PRD.md`. All implementation lives
in `DL_XAI.ipynb` at the repo root.

## How to Work Efficiently (low context — this is the DEFAULT, no need to be told)
- The brain is **queried, not loaded**. Never read whole files or the whole `.agents/` tree "to get context."
- Lookup order for ANY task: (1) the ONE relevant `.agents/` file the task scope points to below, (2) at most 2–3 targeted reads (e.g. the relevant PRD phase section, or specific notebook cells). Full-file reads are the last resort.
- Pull ONLY the `.agents/` file the task scope points to — never preload all of them.
- This runs automatically for every task; the user does NOT have to say "use the second brain."

## Agent Routing Instructions
To prevent context dilution, general invariants and rules are split into modular guides. **Always read these files first based on the scope of your task:**

1.  **Identity, Dev Persona & Response Conventions**:
    *   Location: `.agents/context/identity.md`
    *   Read when: Starting a new session, or before implementing/reviewing a PRD phase (sets the "implement the phase spec exactly" convention).
2.  **Invariants, Tech Stack & File Map**:
    *   Location: `.agents/context/stack-and-rules.md`
    *   Read when: Touching MC Dropout inference, uncertainty thresholds, the XAI assignment policy, DiCE immutable features, or figuring out where a piece of code/data actually lives.
3.  **Historical Decisions**:
    *   Location: `.agents/decisions/log.md`
    *   Read when: Asked "why this dataset / why this UQ method / why this XAI assignment" or considering changing a PRD-frozen choice.
4.  **Active Roadmap & Technical Debt**:
    *   Location: `.agents/projects/active-backlog.md`
    *   Read when: Checking which PRD phase is next, or resuming work after a break.
5.  **Subsystem Notes & Load-Bearing Gotchas**:
    *   Location: `.agents/context/subsystem-notes.md`
    *   Read when: Debugging notebook output-path mismatches, DiCE failures, or anything that looks like a bug but might be expected PRD behavior.

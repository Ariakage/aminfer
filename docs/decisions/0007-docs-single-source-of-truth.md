# ADR-0007: Documentation taxonomy and single sources of truth

- Status: Accepted
- Date: 2026-11-20
- Source: user requirement (init session, "new requirements" 4) and
  `AGENTS.md`

## Context

Progress notes and "what works" lists drift when they are scattered across
many documents.

## Decision

1. `docs/` is split by kind: `docs/architecture/` (architecture design),
   `docs/decisions/` (ADRs) and `docs/guides/` (build and environment guides).
   Nothing else goes in the `docs/` root except the two files below.
2. All project progress converges on `docs/process.md` — milestones, spikes,
   task lists, verification log and the session handoff snapshot.
3. All implemented, unimplemented and planned capabilities converge on
   `docs/status.md`.
4. Other documents link to those two files instead of restating status or
   progress. ADR status values describe decisions, not capabilities.

## Consequences

- A change that alters a capability updates `docs/status.md` in the same
  commit; a finished task updates `docs/process.md`.
- Architecture documents describe design only.

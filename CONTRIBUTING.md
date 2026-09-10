# Contributing / working notes

## Before adding a script, check for existing tooling first

This repo has been assembled from several rounds of ad-hoc debugging
sessions, each producing scripts named after that session's experiment ID
(`A01`, `A08`, etc.). It is easy to end up with two versions of the same
tool — a mature generic one and a throwaway hardcoded one — because the
throwaway one was written without checking `scripts/` first.

**Concrete incident this rule exists because of:** `scripts/server_supervisor.py`
already existed as a generic, model-size-aware process supervisor
(`--model-gib`/`--draft-gib`/`--kv-gib`/`--graph-gib` reserve flags,
status-file reconciliation for dropped SSH sessions) when a later session
pulled a different, worse, hardcoded-for-one-model supervisor script off a
remote box and committed it as a *second* supervisor under a different
name. Both existed in the repo simultaneously with no indication either
one was the "real" one, until this was caught and the redundant copy was
deleted.

**Rule:** before writing or copying in a new script that does something
resembling "supervise a process," "launch a server," "benchmark
throughput," or "inspect a checkpoint," `grep`/`ls` `scripts/`, `tests/`,
and `benchmarks/` first. If something close already exists:
- Extend/parametrize it instead of duplicating it, or
- If the existing one is genuinely inferior, replace it and delete the
  old one in the same commit — don't leave both.

## Naming

Scripts tied to a specific numbered experiment (as tracked in a
model-recipe repo's `STATE.md`) belong in that recipe repo under a name
like `scripts/A08_native_mtp_graphs.sh`, where the numeric ID is a real,
documented cross-reference. Scripts in *this* repo (the generic plugin)
should be named for what they do, not for which session produced them —
if you're about to name something `A0N_whatever.py` here, it probably
belongs in a recipe repo instead, not here.

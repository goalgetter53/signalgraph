# ADR 0001: DuckDB as the single store, with replace-on-write derived tables

- Status: accepted
- Date: 2026-11-19

## Context

signalgraph ingests a few thousand to a few hundred thousand signals per run,
then runs set-oriented work over all of them (entity resolution over every
mention, scoring over every account, ranked exports). It must run on a laptop
and in CI with no services, and re-running any step must not duplicate data.

Options considered:

| Option | For | Against |
|---|---|---|
| Postgres | concurrent writers, mature upserts | a server to run in CI and locally; overkill for one batch writer |
| SQLite | zero-ops, ubiquitous | row-store; weak for the analytical scans scoring/export do; no `list()` aggregates |
| **DuckDB** | zero-ops file, columnar, fast scans, `ON CONFLICT` upserts, JSON, `GREATEST` | single writer process; not a multi-user OLTP database |
| Parquet files | cheap, portable | no upserts; idempotency would be hand-rolled |

## Decision

Use one DuckDB file. Raw inputs (`signals`, `mentions`) are **upserted** on
deterministic keys (`event_id = sha256(source, native_id)`,
`mention_id = sha256(source, name, domain, country)`). Derived outputs
(`accounts`, `account_members`, `scores`) are **replaced inside a
transaction** on every resolve/score run. Watermarks only move forward
(`GREATEST`). Each run is logged in `runs` with its stats.

## Consequences

- Running any command twice yields identical tables (proved by
  `tests/integration/test_pipeline.py::test_running_twice_adds_nothing`).
- Full re-resolution is cheap at this scale (50k mentions resolve in ~12 s, see
  README benchmarks), so incremental ER is deferred (roadmap).
- One writer at a time. The API opens a cursor per request for reads; a
  multi-writer deployment would move raw tables to Postgres and keep DuckDB
  for the analytical side.

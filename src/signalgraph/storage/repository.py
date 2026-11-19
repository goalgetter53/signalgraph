"""DuckDB repository: the only module that writes SQL.

Idempotency rules:
* signals are keyed by ``event_id = hash(source, native_id)``; re-ingesting the
  same item updates mutable fields and ``last_seen_at`` but never adds a row,
* mentions are keyed by a hash of (source, name, domain, country),
* resolve/score *replace* their output tables inside one transaction, so a
  re-run over unchanged input produces byte-identical tables,
* watermarks only move forward (``GREATEST``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from signalgraph.domain.models import (
    Account,
    CompanyMention,
    ScoreBreakdown,
    SignalEvent,
    SignalType,
    Source,
)

SCHEMA = Path(__file__).with_name("schema.sql")


def _naive_utc(ts: datetime) -> datetime:
    return ts.astimezone(UTC).replace(tzinfo=None) if ts.tzinfo else ts


def _aware(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


class Repository:
    def __init__(self, path: Path | str = ":memory:", *, connection: Any = None):
        self.path = str(path)
        if connection is not None:
            self.con = connection
            return
        self.con = duckdb.connect(self.path)
        self.con.execute(SCHEMA.read_text(encoding="utf-8"))

    def cursor(self) -> Repository:
        """A repository on a new cursor of the same database (one per thread/request)."""
        return Repository(self.path, connection=self.con.cursor())

    def close(self) -> None:
        self.con.close()

    # ----------------------------------------------------------------- runs
    def start_run(self, run_id: str, kind: str) -> None:
        self.con.execute(
            "INSERT INTO runs (run_id, kind, started_at) VALUES (?, ?, ?)",
            [run_id, kind, _naive_utc(datetime.now(UTC))],
        )

    def finish_run(self, run_id: str, stats: dict[str, Any]) -> None:
        self.con.execute(
            "UPDATE runs SET finished_at = ?, stats = ? WHERE run_id = ?",
            [_naive_utc(datetime.now(UTC)), json.dumps(stats, default=str), run_id],
        )

    def runs(self) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT run_id, kind, started_at, finished_at, stats FROM runs ORDER BY started_at"
        ).fetchall()
        return [
            {
                "run_id": r[0],
                "kind": r[1],
                "started_at": r[2],
                "finished_at": r[3],
                "stats": json.loads(r[4]) if r[4] else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------ watermarks
    def get_watermark(self, key: str) -> datetime | None:
        row = self.con.execute(
            "SELECT watermark FROM watermarks WHERE connector_key = ?", [key]
        ).fetchone()
        return _aware(row[0]) if row else None

    def advance_watermark(self, key: str, value: datetime) -> None:
        now = _naive_utc(datetime.now(UTC))
        self.con.execute(
            """
            INSERT INTO watermarks (connector_key, watermark, updated_at) VALUES (?, ?, ?)
            ON CONFLICT (connector_key) DO UPDATE SET
                watermark = GREATEST(watermarks.watermark, excluded.watermark),
                updated_at = excluded.updated_at
            """,
            [key, _naive_utc(value), now],
        )

    def watermarks(self) -> dict[str, datetime]:
        rows = self.con.execute("SELECT connector_key, watermark FROM watermarks").fetchall()
        return {k: _aware(v) for k, v in rows}

    # --------------------------------------------------------------- signals
    def upsert_signals(self, events: Sequence[SignalEvent], run_id: str) -> dict[str, int]:
        """Insert new events, refresh existing ones. Returns inserted/updated counts."""
        if not events:
            return {"inserted": 0, "updated": 0}
        unique = {e.event_id: e for e in events}
        ids = list(unique)
        existing = {
            r[0]
            for r in self.con.execute(
                "SELECT event_id FROM signals WHERE event_id IN (SELECT UNNEST(?))", [ids]
            ).fetchall()
        }
        now = _naive_utc(datetime.now(UTC))
        mentions = {e.event_id: CompanyMention.from_event(e) for e in unique.values()}
        rows = [
            [
                e.event_id,
                e.source.value,
                e.native_id,
                e.signal_type.value,
                e.subtype,
                e.company_name,
                e.company_domain,
                e.location,
                e.country,
                _naive_utc(e.occurred_at),
                e.title,
                e.url,
                e.magnitude,
                json.dumps(e.attributes, sort_keys=True, default=str),
                mentions[e.event_id].mention_id,
                run_id,
                now,
            ]
            for e in unique.values()
        ]
        self.con.begin()
        try:
            self.con.executemany(
                """
                INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (event_id) DO UPDATE SET
                    subtype = excluded.subtype, title = excluded.title, url = excluded.url,
                    magnitude = excluded.magnitude, attributes = excluded.attributes,
                    location = excluded.location, last_seen_at = excluded.last_seen_at
                """,
                rows,
            )
            self._upsert_mentions(mentions.values())
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise
        return {"inserted": len(ids) - len(existing), "updated": len(existing)}

    def _upsert_mentions(self, mentions: Iterable[CompanyMention]) -> None:
        unique = {m.mention_id: m for m in mentions}
        self.con.executemany(
            "INSERT INTO mentions VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [
                [
                    m.mention_id,
                    m.source,
                    m.raw_name,
                    m.raw_domain,
                    m.location,
                    m.country,
                    json.dumps(list(m.aliases)),
                ]
                for m in unique.values()
            ],
        )

    def count(self, table: str) -> int:
        if table not in {"signals", "mentions", "accounts", "account_members", "scores", "runs"}:
            raise ValueError(table)
        row = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0

    def mentions(self) -> list[CompanyMention]:
        rows = self.con.execute(
            "SELECT mention_id, source, raw_name, raw_domain, location, country, aliases "
            "FROM mentions ORDER BY mention_id"
        ).fetchall()
        return [
            CompanyMention(
                mention_id=r[0],
                source=r[1],
                raw_name=r[2],
                raw_domain=r[3],
                location=r[4],
                country=r[5],
                aliases=tuple(json.loads(r[6] or "[]")),
            )
            for r in rows
        ]

    _SIGNAL_COLS = (
        "s.source, s.native_id, s.signal_type, s.subtype, s.company_name, s.company_domain, "
        "s.location, s.country, s.occurred_at, s.title, s.url, s.magnitude, s.attributes"
    )

    @staticmethod
    def _row_to_event(r: Sequence[Any]) -> SignalEvent:
        return SignalEvent(
            source=Source(r[0]),
            native_id=r[1],
            signal_type=SignalType(r[2]),
            subtype=r[3],
            company_name=r[4],
            company_domain=r[5],
            location=r[6],
            country=r[7],
            occurred_at=_aware(r[8]),
            title=r[9],
            url=r[10],
            magnitude=r[11],
            attributes=json.loads(r[12]) if r[12] else {},
        )

    def signals_by_account(self) -> dict[str, list[SignalEvent]]:
        rows = self.con.execute(
            f"SELECT m.account_id, {self._SIGNAL_COLS} FROM signals s "
            "JOIN account_members m USING (mention_id) ORDER BY s.occurred_at DESC, s.event_id"
        ).fetchall()
        out: dict[str, list[SignalEvent]] = {}
        for r in rows:
            out.setdefault(r[0], []).append(self._row_to_event(r[1:]))
        return out

    def signals_for_account(self, account_id: str, limit: int = 200) -> list[SignalEvent]:
        rows = self.con.execute(
            f"SELECT {self._SIGNAL_COLS} FROM signals s "
            "JOIN account_members m USING (mention_id) WHERE m.account_id = ? "
            "ORDER BY s.occurred_at DESC, s.event_id LIMIT ?",
            [account_id, limit],
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def list_signals(
        self, signal_type: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[SignalEvent]:
        where, params = ("WHERE s.signal_type = ?", [signal_type]) if signal_type else ("", [])
        rows = self.con.execute(
            f"SELECT {self._SIGNAL_COLS} FROM signals s {where} "
            "ORDER BY s.occurred_at DESC, s.event_id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # -------------------------------------------------------------- accounts
    def replace_accounts(self, accounts: Sequence[Account]) -> None:
        self.con.begin()
        try:
            self.con.execute("DELETE FROM account_members")
            self.con.execute("DELETE FROM accounts")
            self.con.executemany(
                "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?)",
                [
                    [
                        a.account_id,
                        a.name,
                        a.domain,
                        a.country,
                        json.dumps(a.aliases),
                        len(a.member_mention_ids),
                    ]
                    for a in accounts
                ],
            )
            self.con.executemany(
                "INSERT INTO account_members VALUES (?, ?)",
                [[m, a.account_id] for a in accounts for m in a.member_mention_ids],
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def accounts(self) -> list[Account]:
        rows = self.con.execute(
            "SELECT a.account_id, a.name, a.domain, a.country, a.aliases, "
            "list(m.mention_id ORDER BY m.mention_id) FROM accounts a "
            "JOIN account_members m USING (account_id) "
            "GROUP BY ALL ORDER BY a.account_id"
        ).fetchall()
        return [
            Account(
                account_id=r[0],
                name=r[1],
                domain=r[2],
                country=r[3],
                aliases=json.loads(r[4] or "[]"),
                member_mention_ids=list(r[5]),
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- scores
    def replace_scores(self, breakdowns: Sequence[ScoreBreakdown]) -> None:
        self.con.begin()
        try:
            self.con.execute("DELETE FROM scores")
            self.con.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    [
                        b.account_id,
                        b.score,
                        b.raw_points,
                        json.dumps(b.by_type),
                        b.icp_fit,
                        json.dumps(b.icp_reasons),
                        b.model_dump_json(),
                        _naive_utc(b.as_of),
                    ]
                    for b in breakdowns
                ],
            )
            self.con.commit()
        except Exception:
            self.con.rollback()
            raise

    def ranked_accounts(
        self, *, icp_only: bool = False, min_score: float = 0.0, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        rows = self.con.execute(
            """
            SELECT a.account_id, a.name, a.domain, a.country, a.member_count,
                   s.score, s.icp_fit, s.by_type, s.icp_reasons,
                   (SELECT COUNT(*) FROM signals x JOIN account_members m USING (mention_id)
                     WHERE m.account_id = a.account_id) AS n_signals
            FROM accounts a JOIN scores s USING (account_id)
            WHERE s.score >= ? AND (NOT ? OR s.icp_fit)
            ORDER BY s.score DESC, a.account_id
            LIMIT ? OFFSET ?
            """,
            [min_score, icp_only, limit, offset],
        ).fetchall()
        return [
            {
                "account_id": r[0],
                "name": r[1],
                "domain": r[2],
                "country": r[3],
                "member_count": r[4],
                "score": r[5],
                "icp_fit": r[6],
                "by_type": json.loads(r[7] or "{}"),
                "icp_reasons": json.loads(r[8] or "[]"),
                "n_signals": r[9],
            }
            for r in rows
        ]

    def breakdown(self, account_id: str) -> ScoreBreakdown | None:
        row = self.con.execute(
            "SELECT breakdown FROM scores WHERE account_id = ?", [account_id]
        ).fetchone()
        return ScoreBreakdown.model_validate_json(row[0]) if row else None

    def account(self, account_id: str) -> Account | None:
        row = self.con.execute(
            "SELECT a.account_id, a.name, a.domain, a.country, a.aliases, "
            "list(m.mention_id ORDER BY m.mention_id) FROM accounts a "
            "JOIN account_members m USING (account_id) WHERE a.account_id = ? GROUP BY ALL",
            [account_id],
        ).fetchone()
        if row is None:
            return None
        return Account(
            account_id=row[0],
            name=row[1],
            domain=row[2],
            country=row[3],
            aliases=json.loads(row[4] or "[]"),
            member_mention_ids=list(row[5]),
        )

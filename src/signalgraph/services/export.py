"""Exports: flat CSV, Clay webhook rows, and HubSpot companies batch-upsert payload.

Clay: a table's "Webhook" source accepts one JSON object per POST; we emit a
list of flat objects (one per account) the caller can POST row by row.

HubSpot: ``POST /crm/v3/objects/companies/batch/upsert`` takes
``{"inputs": [{"idProperty": ..., "id": ..., "properties": {...}}]}`` (max 100
inputs per request, so we chunk). Upsert needs a *unique-value* property as
``idProperty``; we use a custom ``signalgraph_account_id`` property (create it
with ``hasUniqueValue: true``) so re-exports update instead of duplicating.
``domain`` is still sent so HubSpot can associate contacts. Accounts with no
domain are listed in ``without_domain`` so a RevOps reviewer can decide.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Any

CSV_FIELDS = (
    "account_id",
    "name",
    "domain",
    "country",
    "score",
    "icp_fit",
    "n_signals",
    "funding_points",
    "hiring_points",
    "eng_activity_points",
    "news_points",
    "top_signal",
)


def _flat(row: dict[str, Any], top: str | None) -> dict[str, Any]:
    by_type = row.get("by_type") or {}
    return {
        "account_id": row["account_id"],
        "name": row["name"],
        "domain": row.get("domain") or "",
        "country": row.get("country") or "",
        "score": round(float(row["score"]), 2),
        "icp_fit": bool(row["icp_fit"]),
        "n_signals": int(row.get("n_signals", 0)),
        "funding_points": round(float(by_type.get("funding", 0.0)), 3),
        "hiring_points": round(float(by_type.get("hiring", 0.0)), 3),
        "eng_activity_points": round(float(by_type.get("eng_activity", 0.0)), 3),
        "news_points": round(float(by_type.get("news", 0.0)), 3),
        "top_signal": top or "",
    }


def to_csv(rows: Sequence[dict[str, Any]], top_signals: dict[str, str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(_flat(row, top_signals.get(row["account_id"])))
    return buf.getvalue()


def to_clay_rows(
    rows: Sequence[dict[str, Any]], top_signals: dict[str, str]
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flat = _flat(row, top_signals.get(row["account_id"]))
        flat["source"] = "signalgraph"
        flat["icp_reasons"] = "; ".join(row.get("icp_reasons") or [])
        out.append(flat)
    return out


def to_hubspot_batches(
    rows: Sequence[dict[str, Any]], top_signals: dict[str, str], batch_size: int = 100
) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    without_domain: list[str] = []
    for row in rows:
        if not row.get("domain"):
            without_domain.append(row["account_id"])
        flat = _flat(row, top_signals.get(row["account_id"]))
        inputs.append(
            {
                "idProperty": "signalgraph_account_id",
                "id": row["account_id"],
                "properties": {
                    "domain": row.get("domain") or "",
                    "name": row["name"],
                    "country": row.get("country") or "",
                    "signalgraph_score": str(flat["score"]),
                    "signalgraph_icp_fit": "true" if flat["icp_fit"] else "false",
                    "signalgraph_top_signal": flat["top_signal"][:250],
                    "signalgraph_account_id": row["account_id"],
                },
            }
        )
    batches = [{"inputs": inputs[i : i + batch_size]} for i in range(0, len(inputs), batch_size)]
    return {
        "endpoint": "POST /crm/v3/objects/companies/batch/upsert",
        "batches": batches,
        "without_domain": without_domain,
    }

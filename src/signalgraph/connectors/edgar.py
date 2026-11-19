"""SEC EDGAR Form D (Regulation D private placement) filings -> funding signals.

Discovery uses the public EDGAR full-text search JSON (``efts.sec.gov``), which
returns filing metadata. For each filing we optionally fetch the Form D
``primary_doc.xml`` to read offering size, industry group and *previous
names* (rebrands are an entity-resolution gift). Pooled investment funds are
kept but tagged ``form_d_fund`` so ICP rules can drop them: a VC fund raising
is not an operating company buying software.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any

from defusedxml import ElementTree

from signalgraph.connectors.base import Connector
from signalgraph.connectors.http import HttpError, PoliteClient
from signalgraph.domain.models import SignalEvent, SignalType, Source

log = logging.getLogger(__name__)

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/primary_doc.xml"
_CIK_SUFFIX = re.compile(r"\s*\(CIK \d+\)\s*$")
PAGE_SIZE = 100


def clean_display_name(display: str) -> str:
    return _CIK_SUFFIX.sub("", display).strip()


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None  # e.g. "Indefinite"


def parse_form_d(xml_text: str) -> dict[str, Any]:
    """Extract the fields we use from a Form D ``primary_doc.xml``."""
    root = ElementTree.fromstring(xml_text)

    def text(path: str) -> str | None:
        node = root.find(path)
        return node.text.strip() if node is not None and node.text else None

    previous = [
        v.text.strip()
        for v in root.findall("./primaryIssuer/issuerPreviousNameList/value")
        if v.text and v.text.strip().lower() not in {"none", "n/a"}
    ]
    return {
        "entity_name": text("./primaryIssuer/entityName"),
        "city": text("./primaryIssuer/issuerAddress/city"),
        "state": text("./primaryIssuer/issuerAddress/stateOrCountry"),
        "industry_group": text("./offeringData/industryGroup/industryGroupType"),
        "is_amendment": (text("./offeringData/typeOfFiling/newOrAmendment/isAmendment") or "")
        == "true",
        "total_offering_amount": _to_float(
            text("./offeringData/offeringSalesAmounts/totalOfferingAmount")
        ),
        "total_amount_sold": _to_float(text("./offeringData/offeringSalesAmounts/totalAmountSold")),
        "previous_names": previous,
    }


class EdgarFormDConnector(Connector):
    source_name = "edgar"

    def __init__(
        self,
        client: PoliteClient,
        *,
        lookback_days: int = 7,
        max_filings: int = 50,
        fetch_details: bool = True,
        today: date | None = None,
    ) -> None:
        super().__init__(client)
        self.lookback_days = lookback_days
        self.max_filings = max_filings
        self.fetch_details = fetch_details
        self.today = today

    @property
    def key(self) -> str:
        return "edgar:form_d"

    async def fetch(self, since: datetime | None) -> list[SignalEvent]:
        end = self.today or datetime.now(UTC).date()
        start = since.date() if since else end - timedelta(days=self.lookback_days)
        hits: list[dict[str, Any]] = []
        offset = 0
        while len(hits) < self.max_filings:
            payload = await self.client.get_json(
                SEARCH_URL,
                params={
                    "forms": "D",
                    "dateRange": "custom",
                    "startdt": start.isoformat(),
                    "enddt": end.isoformat(),
                    "from": offset,
                },
            )
            page = payload.get("hits", {}).get("hits", [])
            hits.extend(page)
            total = payload.get("hits", {}).get("total", {}).get("value", 0)
            offset += len(page)
            if not page or offset >= total or len(page) < PAGE_SIZE:
                break
        events = [await self._to_event(h["_source"]) for h in hits[: self.max_filings]]
        return self.keep([e for e in events if e is not None], since)

    async def _to_event(self, src: dict[str, Any]) -> SignalEvent | None:
        names = src.get("display_names") or []
        ciks = src.get("ciks") or []
        adsh = src.get("adsh")
        if not names or not ciks or not adsh:
            return None
        name = clean_display_name(names[0])
        cik = str(int(ciks[0]))
        adsh_nodash = adsh.replace("-", "")
        details: dict[str, Any] = {}
        if self.fetch_details:
            try:
                xml = await self.client.get_text(ARCHIVE_URL.format(cik=cik, adsh=adsh_nodash))
                details = parse_form_d(xml)
            except (HttpError, ElementTree.ParseError, ValueError) as exc:
                log.warning("form D detail fetch failed for %s: %s", adsh, exc)
        industry = details.get("industry_group") or ""
        is_fund = "pooled investment fund" in industry.lower()
        amount = details.get("total_amount_sold") or details.get("total_offering_amount") or 0.0
        location = (src.get("biz_locations") or [None])[0]
        return SignalEvent(
            source=Source.EDGAR,
            native_id=adsh,
            signal_type=SignalType.FUNDING,
            subtype="form_d_fund" if is_fund else "form_d",
            company_name=details.get("entity_name") or name,
            company_domain=None,  # Form D has no website field
            location=location,
            country="US" if src.get("biz_states") else None,
            occurred_at=datetime.fromisoformat(src["file_date"]).replace(tzinfo=UTC),
            title=f"Form D filed by {name}" + (f" ({industry})" if industry else ""),
            url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/",
            magnitude=float(amount),
            attributes={
                "cik": cik,
                "industry_group": industry or None,
                "is_amendment": bool(details.get("is_amendment")) or src.get("file_type") == "D/A",
                "total_offering_amount": details.get("total_offering_amount"),
                "total_amount_sold": details.get("total_amount_sold"),
                "aliases": details.get("previous_names", []),
                "inc_states": src.get("inc_states") or [],
            },
        )

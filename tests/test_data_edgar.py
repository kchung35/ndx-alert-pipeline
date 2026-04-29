"""Tests for SEC EDGAR filing-window handling."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src import data_edgar


def test_list_form4_filings_uses_as_of_window(monkeypatch):
    class FakeResponse:
        def json(self):
            return {
                "filings": {
                    "recent": {
                        "form": ["4", "4", "4", "3", "4"],
                        "accessionNumber": ["old", "inside", "future", "not4", "edge"],
                        "filingDate": [
                            "2026-03-01",
                            "2026-04-15",
                            "2026-04-25",
                            "2026-04-10",
                            "2026-03-22",
                        ],
                        "primaryDocument": ["old.xml", "inside.xml", "future.xml", "not4.xml", "edge.xml"],
                    },
                },
            }

    monkeypatch.setattr(data_edgar, "_get", lambda _url: FakeResponse())

    filings = data_edgar.list_form4_filings("123", lookback_days=30, as_of=date(2026, 4, 21))

    assert [f["accession"] for f in filings] == ["inside", "edge"]


def _row(accession: str, ticker: str = "TEST") -> dict:
    return {
        "ticker": ticker,
        "cik": "123",
        "accession": accession,
        "filing_date": "2026-04-15",
        "transaction_date": "2026-04-15",
        "insider_name": "Jane Doe",
        "position": "CFO",
        "tx_code": "P",
        "is_derivative": False,
        "is_10b5_1": False,
        "shares": 10.0,
        "price": 20.0,
        "value": 200.0,
        "signal_label": "Open Market Purchase",
        "signal_weight": 2,
        "officer_weight": 1.2,
    }


def test_fetch_form4_for_ticker_skips_cached_accessions(monkeypatch, tmp_path):
    monkeypatch.setattr(data_edgar, "FORM4_DIR", tmp_path)
    pd.DataFrame([_row("cached")]).to_parquet(tmp_path / "TEST.parquet", index=False)
    monkeypatch.setattr(data_edgar, "ticker_to_cik", lambda _ticker: "123")
    monkeypatch.setattr(
        data_edgar,
        "list_form4_filings",
        lambda *_args, **_kwargs: [
            {"accession": "cached", "filing_date": "2026-04-10", "primary_doc": "cached.xml"},
            {"accession": "new", "filing_date": "2026-04-15", "primary_doc": "new.xml"},
        ],
    )
    fetched = []

    def fake_fetch(_cik, accession, _primary_doc):
        fetched.append(accession)
        return b"<?xml version='1.0'?><ownershipDocument/>"

    monkeypatch.setattr(data_edgar, "_fetch_form4_xml", fake_fetch)
    monkeypatch.setattr(
        data_edgar,
        "parse_form4_xml",
        lambda _xml, ticker, _cik, accession, _filing_date: [_row(accession, ticker)],
    )

    df = data_edgar.fetch_form4_for_ticker("TEST", as_of=date(2026, 4, 21))

    assert fetched == ["new"]
    assert df["accession"].tolist() == ["new"]


def test_fetch_form4_for_ticker_can_force_refetch_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(data_edgar, "FORM4_DIR", tmp_path)
    pd.DataFrame([_row("cached")]).to_parquet(tmp_path / "TEST.parquet", index=False)
    monkeypatch.setattr(data_edgar, "ticker_to_cik", lambda _ticker: "123")
    monkeypatch.setattr(
        data_edgar,
        "list_form4_filings",
        lambda *_args, **_kwargs: [
            {"accession": "cached", "filing_date": "2026-04-10", "primary_doc": "cached.xml"},
        ],
    )
    fetched = []
    monkeypatch.setattr(
        data_edgar,
        "_fetch_form4_xml",
        lambda _cik, accession, _doc: fetched.append(accession) or b"<?xml version='1.0'?><ownershipDocument/>",
    )
    monkeypatch.setattr(
        data_edgar,
        "parse_form4_xml",
        lambda _xml, ticker, _cik, accession, _filing_date: [_row(accession, ticker)],
    )

    df = data_edgar.fetch_form4_for_ticker(
        "TEST",
        as_of=date(2026, 4, 21),
        skip_cached=False,
    )

    assert fetched == ["cached"]
    assert df["accession"].tolist() == ["cached"]


def test_no_new_form4_run_preserves_existing_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data_edgar, "FORM4_DIR", tmp_path)
    path = tmp_path / "TEST.parquet"
    pd.DataFrame([_row("cached")]).to_parquet(path, index=False)
    monkeypatch.setattr(data_edgar, "ticker_to_cik", lambda _ticker: "123")
    monkeypatch.setattr(
        data_edgar,
        "list_form4_filings",
        lambda *_args, **_kwargs: [
            {"accession": "cached", "filing_date": "2026-04-10", "primary_doc": "cached.xml"},
        ],
    )
    monkeypatch.setattr(
        data_edgar,
        "_fetch_form4_xml",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cached XML fetched")),
    )

    df = data_edgar.fetch_form4_for_ticker("TEST", as_of=date(2026, 4, 21))

    assert df.empty
    assert pd.read_parquet(path)["accession"].tolist() == ["cached"]

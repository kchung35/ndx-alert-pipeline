"""Smoke tests for static dashboard data wiring."""

from __future__ import annotations

from datetime import date

from scripts.generate_dashboard_html import build_real_data_iife


def test_baked_iife_exposes_performance_risk_payload():
    payload = {
        "risk": {"summary": {"n_days": 20}, "equity": [], "drawdown": []},
        "validation": {
            "status": "BLOCKED",
            "usable_days": 2,
            "min_days": 20,
        },
    }

    iife = build_real_data_iife(
        as_of=date(2026, 4, 29),
        alerts=[],
        prices={},
        chains={},
        form4={},
        performance_risk_by_date={"2026-04-29": payload},
        dates_available=["2026-04-29"],
    )

    assert '"PERFORMANCE_RISK"' in iife
    assert "getPerformanceRisk" in iife
    assert '"n_days":20' in iife
    assert '"validation"' in iife


def test_baked_iife_does_not_reuse_latest_performance_for_missing_dates():
    iife = build_real_data_iife(
        as_of=date(2026, 4, 29),
        alerts=[],
        prices={},
        chains={},
        form4={},
        performance_risk_by_date={
            "2026-04-29": {
                "risk": {"summary": {"n_days": 20}, "equity": [], "drawdown": []},
                "validation": {"status": "BLOCKED"},
            },
            "2026-04-21": {
                "risk": {"summary": {"n_days": 10}, "equity": [], "drawdown": []},
                "validation": {"status": "BLOCKED"},
            },
        },
        dates_available=["2026-04-29", "2026-04-21"],
    )

    assert '"2026-04-21":{"risk":{"summary":{"n_days":10}' in iife
    assert "BAKED.PERFORMANCE_RISK[key] || null" in iife
    assert "BAKED.PERFORMANCE_RISK[dateStr] || BAKED.PERFORMANCE_RISK[BAKED.AS_OF]" not in iife

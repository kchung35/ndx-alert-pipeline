from __future__ import annotations

from datetime import date, datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path

from src.newsletter_export import (
    build_eml, export_newsletter, load_newsletter_context,
    render_newsletter_html, render_newsletter_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_TS = datetime(2026, 4, 29, 13, 45, 0, tzinfo=timezone.utc)


def test_newsletter_render_contains_snapshot_kpis_and_quality():
    ctx = load_newsletter_context(
        date(2026, 4, 21),
        project_root=PROJECT_ROOT,
        generated_at=FIXED_TS,
    )
    html = render_newsletter_html(ctx)
    text = render_newsletter_text(ctx)

    assert "NDX Alert Snapshot" in html
    assert "2026-04-21" in html
    assert "generated 2026-04-29 13:45:00 UTC" in html
    assert "Long alerts" in html
    assert "Short alerts" in html
    assert "Momentum long" in html
    assert "Options quality:" in html
    assert "WDC" in text
    assert "Actionable alerts:" in text


def test_eml_generation_has_body_fallbacks_blank_recipient_and_attachments(tmp_path):
    ctx = load_newsletter_context(
        date(2026, 4, 21),
        project_root=PROJECT_ROOT,
        generated_at=FIXED_TS,
    )
    html_body = render_newsletter_html(ctx)
    text_body = render_newsletter_text(ctx)
    attachment = tmp_path / "NDX Alert Desk.html"
    attachment.write_text("<html>snapshot</html>", encoding="utf-8")

    msg = build_eml(
        ctx=ctx,
        html_body=html_body,
        text_body=text_body,
        attachments=[attachment],
    )
    parsed = BytesParser(policy=policy.default).parsebytes(msg.as_bytes())

    assert parsed["To"] == ""
    assert parsed["Subject"] == "NDX Alert Snapshot - 2026-04-21 - generated 2026-04-29 13:45 UTC"
    assert parsed.get_body(preferencelist=("html",)) is not None
    assert parsed.get_body(preferencelist=("plain",)) is not None
    assert [p.get_filename() for p in parsed.iter_attachments()] == ["NDX Alert Desk.html"]


def test_export_newsletter_writes_timestamped_package_without_png(tmp_path):
    artifacts = export_newsletter(
        date(2026, 4, 21),
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        generated_at=FIXED_TS,
        render_png=False,
    )

    assert artifacts.output_dir == tmp_path / "2026-04-21_20260429-134500"
    assert artifacts.newsletter_html.exists()
    assert artifacts.newsletter_text.exists()
    assert artifacts.eml.exists()
    assert artifacts.dashboard_html is not None
    assert artifacts.dashboard_html.exists()
    assert artifacts.dashboard_png is None
    assert artifacts.png_status == "skipped: disabled"

    parsed = BytesParser(policy=policy.default).parsebytes(artifacts.eml.read_bytes())
    filenames = {p.get_filename() for p in parsed.iter_attachments()}
    assert {
        "newsletter.html",
        "newsletter.txt",
        "NDX Alert Desk.html",
    }.issubset(filenames)


def test_export_newsletter_png_failure_still_succeeds(tmp_path, monkeypatch):
    import src.newsletter_export as exporter

    def fake_png(_html_path, _png_path, **_kwargs):
        return "skipped: screenshot failed (test)"

    monkeypatch.setattr(exporter, "render_dashboard_png", fake_png)
    artifacts = export_newsletter(
        date(2026, 4, 21),
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
        generated_at=FIXED_TS,
        render_png=True,
    )

    assert artifacts.eml.exists()
    assert artifacts.dashboard_png is None
    assert artifacts.png_status == "skipped: screenshot failed (test)"

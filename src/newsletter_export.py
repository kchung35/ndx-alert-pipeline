"""Timestamped email-newsletter export for the NDX Alert Desk.

The exporter is intentionally local-only: it writes newsletter artifacts and a
draft .eml file, but never sends mail or contacts a third-party service.
"""

from __future__ import annotations

import base64
import mimetypes
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from html import escape
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ALERTS_DIR = DATA_DIR / "alerts"
OPT_SIG_DIR = DATA_DIR / "options_signals"
UNIVERSE_PARQUET = DATA_DIR / "universe.parquet"
STATIC_DASHBOARD = PROJECT_ROOT / "NDX Alert Desk.html"
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "exports" / "newsletters"

ACTIONABLE_TIERS = {
    "STRONG_BULLISH",
    "CONFLUENCE_BULLISH",
    "MOMENTUM_LONG",
    "STRONG_BEARISH",
    "CONFLUENCE_BEARISH",
}
TIER_ORDER = {
    "STRONG_BULLISH": 0,
    "CONFLUENCE_BULLISH": 1,
    "MOMENTUM_LONG": 2,
    "STRONG_BEARISH": 3,
    "CONFLUENCE_BEARISH": 4,
    "NO_ALERT": 5,
}


@dataclass(frozen=True)
class NewsletterContext:
    as_of_date: date
    generated_at_local: datetime
    generated_at_utc: datetime
    alerts: pd.DataFrame
    options_quality_counts: dict[str, int]
    subject: str


@dataclass(frozen=True)
class ExportArtifacts:
    output_dir: Path
    newsletter_html: Path
    newsletter_text: Path
    dashboard_html: Path | None
    dashboard_png: Path | None
    eml: Path
    subject: str
    png_status: str


def _coerce_generated_at(generated_at: datetime | None) -> tuple[datetime, datetime]:
    local = generated_at or datetime.now().astimezone()
    if local.tzinfo is None:
        local = local.astimezone()
    utc = local.astimezone(timezone.utc)
    return local, utc


def _timestamp_slug(generated_at: datetime) -> str:
    return generated_at.strftime("%Y%m%d-%H%M%S")


def _subject(as_of: date, generated_at_local: datetime) -> str:
    stamp = generated_at_local.strftime("%Y-%m-%d %H:%M %Z").strip()
    return f"NDX Alert Snapshot - {as_of.isoformat()} - generated {stamp}"


def _fmt_z(v: object) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):+.2f}"


def _fmt_pct(v: object) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.0%}"


def _fmt_mcap(v: object) -> str:
    if pd.isna(v) or float(v) <= 0:
        return "-"
    b = float(v) / 1e9
    return f"${b:,.1f}B"


def load_newsletter_context(
    as_of: date,
    *,
    project_root: Path = PROJECT_ROOT,
    generated_at: datetime | None = None,
) -> NewsletterContext:
    """Load alert + options data and normalize rows for newsletter rendering."""
    local, utc = _coerce_generated_at(generated_at)
    alerts_path = project_root / "data" / "alerts" / f"{as_of.isoformat()}.parquet"
    if not alerts_path.exists():
        raise FileNotFoundError(f"No alerts parquet found at {alerts_path}")

    alerts = pd.read_parquet(alerts_path).copy()
    universe_path = project_root / "data" / "universe.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        keep = ["ticker", "company", "sector", "market_cap"]
        alerts = alerts.merge(universe[keep], on="ticker", how="left")
    else:
        alerts["company"] = ""
        alerts["sector"] = ""
        alerts["market_cap"] = pd.NA

    opt_path = project_root / "data" / "options_signals" / f"{as_of.isoformat()}.parquet"
    quality_counts: dict[str, int] = {}
    if opt_path.exists():
        options = pd.read_parquet(opt_path)
        if "options_quality" in options.columns:
            quality_counts = {
                str(k): int(v)
                for k, v in options["options_quality"].value_counts().sort_index().items()
            }
        opt_cols = [
            c for c in [
                "ticker", "options_quality", "options_coverage",
                "flow_score", "skew_score", "vol_stress_score",
            ]
            if c in options.columns
        ]
        if opt_cols:
            alerts = alerts.merge(options[opt_cols], on="ticker", how="left")

    return NewsletterContext(
        as_of_date=as_of,
        generated_at_local=local,
        generated_at_utc=utc,
        alerts=alerts,
        options_quality_counts=quality_counts,
        subject=_subject(as_of, local),
    )


def kpi_summary(alerts: pd.DataFrame) -> dict[str, object]:
    """Return dashboard-aligned KPI values for the newsletter header."""
    n_bull = int((alerts["tier"] == "STRONG_BULLISH").sum())
    n_bear = int((alerts["tier"] == "STRONG_BEARISH").sum())
    n_conf_bull = int((alerts["tier"] == "CONFLUENCE_BULLISH").sum())
    n_conf_bear = int((alerts["tier"] == "CONFLUENCE_BEARISH").sum())
    n_mom = int((alerts["tier"] == "MOMENTUM_LONG").sum())
    comp_min = float(alerts["composite"].min()) if len(alerts) else 0.0
    comp_max = float(alerts["composite"].max()) if len(alerts) else 0.0
    return {
        "long_alerts": n_bull + n_conf_bull,
        "short_alerts": n_bear + n_conf_bear,
        "momentum_longs": n_mom,
        "composite_range": f"{comp_min:+.2f} ... {comp_max:+.2f}",
        "universe_count": int(alerts["ticker"].nunique()) if "ticker" in alerts else len(alerts),
    }


def _actionable_rows(alerts: pd.DataFrame) -> pd.DataFrame:
    rows = alerts[alerts["tier"].isin(ACTIONABLE_TIERS)].copy()
    if rows.empty:
        return rows
    rows["_tier_order"] = rows["tier"].map(TIER_ORDER).fillna(99)
    rows["_abs_comp"] = rows["composite"].abs()
    return rows.sort_values(["_tier_order", "_abs_comp"], ascending=[True, False])


def _watchlist_rows(alerts: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    rows = alerts[alerts["tier"] == "NO_ALERT"].copy()
    if rows.empty:
        return rows
    rows["_abs_comp"] = rows["composite"].abs()
    return rows.sort_values("_abs_comp", ascending=False).head(limit)


def _quality_summary(ctx: NewsletterContext) -> str:
    if not ctx.options_quality_counts:
        return "Options quality: unavailable"
    parts = [f"{k} {v}" for k, v in sorted(ctx.options_quality_counts.items())]
    return "Options quality: " + ", ".join(parts)


def render_newsletter_html(ctx: NewsletterContext) -> str:
    """Render email-ready HTML with inline styles."""
    kpis = kpi_summary(ctx.alerts)
    actionable = _actionable_rows(ctx.alerts)
    watch = _watchlist_rows(ctx.alerts)

    def tr(row: pd.Series) -> str:
        return (
            "<tr>"
            f"<td>{escape(str(row['tier']))}</td>"
            f"<td><strong>{escape(str(row['ticker']))}</strong></td>"
            f"<td>{escape(str(row.get('company', '') or ''))}</td>"
            f"<td>{escape(str(row.get('sector', '') or ''))}</td>"
            f"<td style='text-align:right'>{_fmt_z(row.get('composite'))}</td>"
            f"<td style='text-align:right'>{_fmt_z(row.get('factor_z'))}</td>"
            f"<td style='text-align:right'>{_fmt_z(row.get('options_z'))}</td>"
            f"<td style='text-align:right'>{_fmt_z(row.get('insider_z'))}</td>"
            f"<td>{escape(str(row.get('rationale', '') or ''))}</td>"
            "</tr>"
        )

    actionable_html = (
        "".join(tr(r) for _, r in actionable.iterrows())
        if not actionable.empty
        else "<tr><td colspan='9'>No actionable alerts in this snapshot.</td></tr>"
    )
    watch_html = (
        "".join(
            "<tr>"
            f"<td><strong>{escape(str(r['ticker']))}</strong></td>"
            f"<td>{escape(str(r.get('company', '') or ''))}</td>"
            f"<td>{escape(str(r.get('sector', '') or ''))}</td>"
            f"<td style='text-align:right'>{_fmt_z(r.get('composite'))}</td>"
            f"<td style='text-align:right'>{_fmt_z(r.get('options_z'))}</td>"
            f"<td style='text-align:right'>{escape(str(r.get('options_quality', '-') or '-'))}</td>"
            f"<td style='text-align:right'>{_fmt_pct(r.get('options_coverage'))}</td>"
            "</tr>"
            for _, r in watch.iterrows()
        )
        if not watch.empty
        else "<tr><td colspan='7'>No watchlist rows available.</td></tr>"
    )

    generated = ctx.generated_at_local.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    generated_utc = ctx.generated_at_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{escape(ctx.subject)}</title></head>
<body style="margin:0;background:#0A0C10;color:#F0F1F5;font-family:Arial,Helvetica,sans-serif;">
  <div style="max-width:980px;margin:0 auto;padding:24px;">
    <h1 style="margin:0 0 6px 0;font-size:28px;">NDX Alert Snapshot</h1>
    <p style="margin:0 0 18px 0;color:#B4B9C6;">
      As-of <strong>{ctx.as_of_date.isoformat()}</strong> · generated {escape(generated)}
      · UTC {escape(generated_utc)}
    </p>

    <table role="presentation" style="width:100%;border-collapse:collapse;margin:0 0 20px 0;">
      <tr>
        <td style="padding:12px;border:1px solid #2D3240;background:#12151C;">Long alerts<br><strong>{kpis['long_alerts']}</strong></td>
        <td style="padding:12px;border:1px solid #2D3240;background:#12151C;">Short alerts<br><strong>{kpis['short_alerts']}</strong></td>
        <td style="padding:12px;border:1px solid #2D3240;background:#12151C;">Momentum long<br><strong>{kpis['momentum_longs']}</strong></td>
        <td style="padding:12px;border:1px solid #2D3240;background:#12151C;">Composite range<br><strong>{kpis['composite_range']}</strong></td>
      </tr>
    </table>

    <h2 style="font-size:18px;margin:20px 0 8px 0;">Actionable Alerts</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#1A1E27;color:#B4B9C6;">
          <th align="left">Tier</th><th align="left">Ticker</th><th align="left">Company</th>
          <th align="left">Sector</th><th align="right">Composite</th><th align="right">Factor</th>
          <th align="right">Options</th><th align="right">Insider</th><th align="left">Rationale</th>
        </tr>
      </thead>
      <tbody>{actionable_html}</tbody>
    </table>

    <h2 style="font-size:18px;margin:22px 0 8px 0;">Top Watchlist By Absolute Composite</h2>
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="background:#1A1E27;color:#B4B9C6;">
          <th align="left">Ticker</th><th align="left">Company</th><th align="left">Sector</th>
          <th align="right">Composite</th><th align="right">Options z</th>
          <th align="right">Options quality</th><th align="right">Coverage</th>
        </tr>
      </thead>
      <tbody>{watch_html}</tbody>
    </table>

    <p style="margin:22px 0 0 0;color:#B4B9C6;font-size:13px;">
      {_quality_summary(ctx)}. Options score is a quality-gated yfinance snapshot signal;
      the heatmap surface is visualization-only and does not change alert scoring.
    </p>
    <p style="margin:10px 0 0 0;color:#6B7182;font-size:12px;">
      Research prototype only. Not investment advice. Free data: yfinance, SEC EDGAR, Ken French.
    </p>
  </div>
</body>
</html>
"""


def render_newsletter_text(ctx: NewsletterContext) -> str:
    kpis = kpi_summary(ctx.alerts)
    lines = [
        "NDX Alert Snapshot",
        f"As-of: {ctx.as_of_date.isoformat()}",
        f"Generated: {ctx.generated_at_local.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}",
        f"Generated UTC: {ctx.generated_at_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"Long alerts: {kpis['long_alerts']}",
        f"Short alerts: {kpis['short_alerts']}",
        f"Momentum long: {kpis['momentum_longs']}",
        f"Composite range: {kpis['composite_range']}",
        "",
        "Actionable alerts:",
    ]
    actionable = _actionable_rows(ctx.alerts)
    if actionable.empty:
        lines.append("- None")
    else:
        for _, row in actionable.iterrows():
            lines.append(
                "- {ticker} {tier} comp={comp} factor={factor} options={options} "
                "insider={insider} rationale={rationale}".format(
                    ticker=row["ticker"],
                    tier=row["tier"],
                    comp=_fmt_z(row.get("composite")),
                    factor=_fmt_z(row.get("factor_z")),
                    options=_fmt_z(row.get("options_z")),
                    insider=_fmt_z(row.get("insider_z")),
                    rationale=row.get("rationale", ""),
                )
            )
    lines.extend(["", "Top watchlist by absolute composite:"])
    watch = _watchlist_rows(ctx.alerts)
    if watch.empty:
        lines.append("- None")
    else:
        for _, row in watch.iterrows():
            lines.append(
                f"- {row['ticker']} comp={_fmt_z(row.get('composite'))} "
                f"options={_fmt_z(row.get('options_z'))} "
                f"quality={row.get('options_quality', '-') or '-'}"
            )
    lines.extend([
        "",
        _quality_summary(ctx),
        "Options score is a quality-gated yfinance snapshot signal; heatmaps are visualization-only.",
        "Research prototype only. Not investment advice.",
    ])
    return "\n".join(lines) + "\n"


def build_eml(
    *,
    ctx: NewsletterContext,
    html_body: str,
    text_body: str,
    attachments: list[Path],
) -> EmailMessage:
    """Create an email draft message. This function never sends anything."""
    msg = EmailMessage()
    msg["Subject"] = ctx.subject
    msg["To"] = ""
    msg["From"] = ""
    msg["Date"] = format_datetime(ctx.generated_at_utc)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    for path in attachments:
        if path is None or not path.exists():
            continue
        ctype, _encoding = mimetypes.guess_type(str(path))
        if ctype is None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(
            path.read_bytes(),
            maintype=maintype,
            subtype=subtype,
            filename=path.name,
        )
    return msg


def render_dashboard_png(
    html_path: Path,
    png_path: Path,
    *,
    width: int = 1440,
    height: int = 1400,
) -> str:
    """Render a full-page PNG with Playwright if available."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        return f"skipped: Playwright unavailable ({exc})"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(html_path.resolve().as_uri(), wait_until="load", timeout=15000)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
    except Exception as exc:
        if png_path.exists():
            png_path.unlink()
        return f"skipped: screenshot failed ({exc})"
    return "created"


def export_newsletter(
    as_of: date,
    *,
    project_root: Path = PROJECT_ROOT,
    output_root: Path | None = None,
    generated_at: datetime | None = None,
    render_png: bool = True,
) -> ExportArtifacts:
    ctx = load_newsletter_context(
        as_of,
        project_root=project_root,
        generated_at=generated_at,
    )
    export_root = output_root or project_root / "exports" / "newsletters"
    out_dir = export_root / f"{as_of.isoformat()}_{_timestamp_slug(ctx.generated_at_local)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    html_body = render_newsletter_html(ctx)
    text_body = render_newsletter_text(ctx)

    newsletter_html = out_dir / "newsletter.html"
    newsletter_text = out_dir / "newsletter.txt"
    newsletter_html.write_text(html_body, encoding="utf-8")
    newsletter_text.write_text(text_body, encoding="utf-8")

    dashboard_src = project_root / "NDX Alert Desk.html"
    dashboard_copy = None
    if dashboard_src.exists():
        dashboard_copy = out_dir / "NDX Alert Desk.html"
        shutil.copy2(dashboard_src, dashboard_copy)

    png_path = out_dir / "dashboard-snapshot.png"
    png_status = "skipped: disabled"
    dashboard_png = None
    if render_png and dashboard_copy is not None:
        png_status = render_dashboard_png(dashboard_copy, png_path)
        if png_status == "created":
            dashboard_png = png_path

    attachments = [newsletter_html, newsletter_text]
    if dashboard_copy is not None:
        attachments.append(dashboard_copy)
    if dashboard_png is not None:
        attachments.append(dashboard_png)

    eml = out_dir / "NDX Alert Snapshot.eml"
    eml.write_bytes(
        build_eml(
            ctx=ctx,
            html_body=html_body,
            text_body=text_body,
            attachments=attachments,
        ).as_bytes()
    )

    return ExportArtifacts(
        output_dir=out_dir,
        newsletter_html=newsletter_html,
        newsletter_text=newsletter_text,
        dashboard_html=dashboard_copy,
        dashboard_png=dashboard_png,
        eml=eml,
        subject=ctx.subject,
        png_status=png_status,
    )


def attachment_as_data_uri(path: Path) -> str:
    """Small helper for UI download links."""
    ctype, _encoding = mimetypes.guess_type(str(path))
    ctype = ctype or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{ctype};base64,{encoded}"

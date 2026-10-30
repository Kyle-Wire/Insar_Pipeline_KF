"""The dashboard's Results screen (v on a finished project): pick an area and a
date range, get the per-point and area results, export them. The work is
results.analyse / results.export; this is only the form around it. Both run
in worker threads and report a failure on the screen (techout-2: an export
error used to escape the button handler and close the whole dashboard)."""
from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, ProgressBar, Select, Static

CUSTOM = "__custom__"


def ground_word(cfg) -> str:
    """The project's words for its comparison ground, as its client report
    says them (client_report.json control_ground_label: the measured land
    cover, or a control_ground_label someone set; 28 Oct 2026); before a
    report exists, the config's (report_insights.ground_word)."""
    import json
    from .report_insights import ground_word as gw
    try:
        rep = json.loads((Path(str(cfg.project_dir)) / "deliverables" / "client_report.json").read_text())
        if rep.get("control_ground_label"):
            return gw(rep)
    except (AttributeError, TypeError, OSError, ValueError):
        pass
    return gw(cfg)


def _z(text: str) -> str:
    """No '-0.000' on screen (report_insights.no_negative_zero)."""
    from .report_insights import no_negative_zero
    return no_negative_zero(text)


def total_text(s: dict, u: str) -> str:
    """The window's measured total with its 95% range, as the workbook and
    the report give it (results.area_summary measured_change_95pct_range: the
    report's total_range, or for an area that could not be compared with
    farmland the uncertainty that can be stated). F8, 7 Oct 2026: the screen
    printed the total bare beside a ranged rate (E9: a total is never
    printed without its range)."""
    tot = s.get(f"mean_vertical_cumulative_{u}")
    if tot is None:
        return ""
    txt = f"{tot:+.2f} {u} measured over the window"
    trng = s.get(f"measured_change_95pct_range_{u}")
    if trng and None not in trng[:2]:
        return txt + f", 95% range {trng[0]:+.2f} to {trng[1]:+.2f} {u}"
    return txt + ", no 95% range"


def reading_style(s: dict) -> str:
    """The colour of an area reading from its fields, not from words in it:
    detected yellow, within normal variation green, not testable / whole
    area / relative only grey."""
    if s.get("testable") is False or (s.get("kind") == "full" and not s.get("gnss_anchored")):
        return "dim"
    if s.get("kind") == "full":
        return "bold"
    sn = s.get("signal_to_noise")
    if sn is not None and sn >= 1:
        return "bold yellow"
    return "green"


class ResultsScreen(Screen):
    BINDINGS = [("escape", "back", "Back"), ("e", "export", "Export"), ("o", "open", "Open results folder")]

    def __init__(self, config: str):
        super().__init__()
        self.config = config
        self.cfg = None
        self.ctx = None
        self.res = None
        self.last_export: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="rbody"):
            yield Static(id="rtitle")
            with Horizontal(classes="row"):
                yield Select([("Custom area (type below)", CUSTOM)], id="rarea", allow_blank=False)
                yield Input(placeholder="custom area: lat,lon,lat,lon · lat,lon (+2 km) · WKT · polygon file path",
                            id="rcustom")
            with Horizontal(classes="row"):
                yield Static("From", classes="lbl")
                yield Input(placeholder="YYYY-MM-DD", id="rfrom", classes="short")
                yield Static("To", classes="lbl")
                yield Input(placeholder="YYYY-MM-DD", id="rto", classes="short")
                yield Button("Full record", id="rfull")
                yield Button("Calculate", id="rgo", variant="primary")
                yield Button("Export CSV + Excel", id="rexp", disabled=True)
            yield ProgressBar(total=100, id="rprog", show_eta=False)
            yield Static(id="rsummary")
            yield DataTable(id="rstats", cursor_type="none", zebra_stripes=True)
            yield DataTable(id="rannual", cursor_type="none", zebra_stripes=True)
            yield Static(id="rnotes")
        yield Footer()

    def on_mount(self) -> None:
        from . import client_report as cr
        from .config import load_config
        self.cfg = load_config(self.config)
        names = [n for n, _ in cr.parse_areas(self.cfg)]
        sel = self.query_one("#rarea", Select)
        sel.set_options([(n, n) for n in names] + [("Custom area (type below)", CUSTOM)])
        sel.value = next((n for n in names if n.lower() != "full satellite imaging"), names[0] if names else CUSTOM)
        self.query_one("#rtitle", Static).update(Text.assemble(
            (" Results ", "bold reverse"), (f"  {Path(self.config).stem.replace('config_', '')}   ", "bold"),
            ("any area, any date range, in the client report's terms: relative to ordinary farmland, "
             "GNSS-anchored only when this record's GNSS tie is", "dim")))
        self.query_one("#rprog", ProgressBar).display = False
        self._full()

    def _full(self) -> None:
        import json
        rep = Path(self.cfg.project_dir) / "deliverables" / "client_report.json"
        d0, d1 = (json.loads(rep.read_text()).get("dates") or ["", ""]) if rep.exists() else ("", "")
        self.query_one("#rfrom", Input).value = f"{d0[:4]}-{d0[4:6]}-{d0[6:]}" if d0 else ""
        self.query_one("#rto", Input).value = f"{d1[:4]}-{d1[4:6]}-{d1[6:]}" if d1 else ""

    @on(Button.Pressed, "#rfull")
    def full(self) -> None:
        self._full()

    @on(Button.Pressed, "#rgo")
    @on(Input.Submitted)
    def go(self) -> None:
        area = self.query_one("#rarea", Select).value
        if area == CUSTOM:
            area = self.query_one("#rcustom", Input).value.strip()
            if not area:
                self.notify("Type the custom area first.", severity="warning")
                return
        pb = self.query_one("#rprog", ProgressBar)
        pb.display = True
        pb.update(total=100, progress=0)
        self.query_one("#rsummary", Static).update(Text("Reading the time series…", style="cyan"))
        self.query_one("#rexp", Button).disabled = True
        self._run(str(area), self.query_one("#rfrom", Input).value.strip() or None,
                  self.query_one("#rto", Input).value.strip() or None)

    @work(thread=True, exclusive=True, group="results")
    def _run(self, area, start, end) -> None:
        from . import results
        from .panel_history import _Context
        try:
            if self.ctx is None:
                self.ctx = _Context(self.cfg)
            pb = self.query_one("#rprog", ProgressBar)
            res = results.analyse(self.cfg, area, start, end, ctx=self.ctx,
                                  progress=lambda d, n: self.app.call_from_thread(pb.update, total=n, progress=d))
            self.app.call_from_thread(self._show, res)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self._fail, exc)

    def _fail(self, exc) -> None:
        self.query_one("#rprog", ProgressBar).display = False
        self.query_one("#rsummary", Static).update(Text(f"✗ {exc}", style="red"))

    def _show(self, res) -> None:
        self.res = res
        u = res.unit
        s = res.area_summary
        self.query_one("#rprog", ProgressBar).display = False
        rng = s.get(f"vertical_95pct_range_{u}_yr")
        t = Text()
        t.append(f"\n{res.area}", style="bold")
        # radar resolution cells, not independent measurements: neighbouring
        # cells share ground (E25, as the report and the workbook word it)
        n_rc = s.get("radar_resolution_cells", s.get("independent_measurements"))
        t.append(f"   {s['points']:,} points{f' (about {n_rc:,} radar resolution cells)' if n_rc else ''} · "
                 f"{s['images']} images · "
                 f"{s['first_image']} to {s['last_image']} ({s['years']:.2f} years)\n", style="dim")
        v = s.get(f"mean_vertical_velocity_{u}_yr")
        tot = s.get(f"mean_vertical_cumulative_{u}")
        t.append(_z(f"Area, vertical: {v:+.3f} {u}/yr") if v is not None else
                 "Area, vertical: relative only (no GNSS anchor)", style="bold")
        # each range beside its own figure: the rate's, then the total's (F8).
        # An area that could not be compared with farmland gives the range
        # that can be stated (the report's untested range, E9), said so - its
        # total carries one, so its rate no longer reads "no 95% range"
        # (in the project's own word for its comparison ground, as the report)
        urng = None if rng else s.get(f"untested_vertical_95pct_range_{u}_yr")
        ground = ground_word(getattr(self, "cfg", None))
        t.append(_z((f"   95% range {rng[0]:+.3f} to {rng[1]:+.3f} {u}/yr" if rng else
                     f"   95% range {urng[0]:+.3f} to {urng[1]:+.3f} {u}/yr, not compared with {ground}" if urng
                     else "   no 95% range")
                    + (f"   ({total_text(s, u)})" if tot is not None else "") + "\n"))
        snr = s.get("signal_to_noise")
        reading = s.get("reading") or "—"
        t.append(f"Signal ÷ threshold {snr:.1f}: {reading}" if snr is not None else reading, style=reading_style(s))
        det = s.get(f"reliably_detects_{u}_yr")
        if det:
            t.append(f"   (reliably detects {det:.2f} {u}/yr or more, 4 times in 5)", style="dim")
        if s.get("basis"):
            t.append("\n" + s["basis"], style="dim" if s.get("gnss_anchored") else "yellow")
        self.query_one("#rsummary", Static).update(t)
        st = self.query_one("#rstats", DataTable)
        st.clear(columns=True)
        st.add_columns("Across the measurements", "Unit", "Mean", "Min", "Max", "Range", "Std dev", "Median", "n")
        for q, v in res.stats.items():
            if not v.get("n"):
                continue
            unit = f"{u}/yr" if "velocity" in q else u
            st.add_row(Text(q, style="bold"), unit, *[_z(f"{v[k]:+.3f}") if k not in ("range", "std") else f"{v[k]:.3f}"
                                                      for k in ("mean", "min", "max", "range", "std", "median")],
                       f"{v['n']:,}")
        an = self.query_one("#rannual", DataTable)
        an.clear(columns=True)
        an.add_columns("Year", "Images", f"Area ({u}/yr, vertical)", "95% range", "Signal ÷ threshold", "Reading")
        for r in res.annual:
            lo_, hi_ = r[f"area_mean_95pct_low_{u}_yr"], r[f"area_mean_95pct_high_{u}_yr"]
            m_ = r.get(f"area_mean_vertical_{u}_yr")
            rd = r.get("area_reading") or ""
            an.add_row(f"{r['year']}{' (part)' if r['partial_year'] else ''}", str(r["n_dates"]),
                       _z(f"{m_:+.3f}") if m_ is not None else "—",
                       _z(f"{lo_:+.3f} to {hi_:+.3f}") if lo_ is not None else "—",
                       f"{r['area_signal_to_noise']:.1f}" if r.get("area_signal_to_noise") is not None else "—",
                       Text(rd, style=("bold yellow" if rd.startswith("stands out") else
                                       "green" if rd.startswith("within") else "dim")))
        self.query_one("#rnotes", Static).update(Text("\n".join(res.notes), style="dim"))
        self.query_one("#rexp", Button).disabled = False

    @on(Button.Pressed, "#rexp")
    def action_export(self) -> None:
        if self.res is None:
            self.notify("Calculate first.", severity="warning")
            return
        self.query_one("#rexp", Button).disabled = True
        self.notify("Writing the CSV files and the workbook…")
        self._export(self.res)

    @work(thread=True, exclusive=True, group="results-export")
    def _export(self, res) -> None:
        """results.export off the UI thread; a failure is shown, never raised
        into Textual (which would close the dashboard). last_export is set
        only when the export succeeded."""
        from . import results
        try:
            out = results.export(self.cfg, res)
        except Exception as exc:                             # noqa: BLE001
            self.app.call_from_thread(self._export_failed, exc)
            return
        self.app.call_from_thread(self._exported, out)

    def _exported(self, out) -> None:
        self.last_export = out
        self.query_one("#rexp", Button).disabled = False
        self.notify(f"Exported to {out}  (o opens it)")

    def _export_failed(self, exc) -> None:
        self.query_one("#rexp", Button).disabled = False
        self.notify(f"Export failed: {exc}. No workbook was left behind; the area figures above are unchanged.",
                    severity="error", timeout=12)

    def action_open(self) -> None:
        from .files_screen import open_default
        p = self.last_export or (Path(self.cfg.project_dir) / "deliverables" / "results")
        p.mkdir(parents=True, exist_ok=True)
        self.notify(f"Opening {open_default(p)}")

    def action_back(self) -> None:
        self.app.pop_screen()

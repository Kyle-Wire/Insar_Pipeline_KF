"""
Where every source PDF of the T1 digitising came from (8 Oct 2026).

`python sources.py` writes ../sources.json: file name, public URL, size,
sha256, and (for gm3.au files) whether the URL still answers with the same
size (a HEAD request only; nothing is downloaded).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "sources"

GM3 = "https://gm3.au/wp-content/uploads"
TCOL = "https://www.tahmoorcolliery.com.au/wp-content/uploads/sites/8/2022/10"

SOURCES = {
    # Tahmoor West six-monthly subsidence impact reports (Reports 1-8), index https://gm3.au/tahmoor-mine/
    "tahmoor_west_report_1.pdf": f"{GM3}/2026/09/Report-1-–-15-November-2019-to-5-May-2020-for-Longwalls-West-1-and-West-2.pdf",
    "tahmoor_west_report_2.pdf": f"{GM3}/2026/09/Report-2-–-8-April-2020-to-6-November-2020-for-Longwalls-West-1-and-West-2.pdf",
    "tahmoor_west_report_3.pdf": f"{GM3}/2026/09/Report-3-–-13-October-2020-to-21-April-2021-for-Longwalls-West-1-and-West-2.pdf",
    "tahmoor_west_report_4.pdf": f"{GM3}/2026/09/Report-4-–-1-April-2021-to-20-October-2021-for-Longwalls-West-1-to-West-4.pdf",
    "tahmoor_west_report_5.pdf": f"{GM3}/2026/09/Report-5-–-1-October-2021-to-15-May-2022-for-Longwalls-West-1-to-West-4.pdf",
    "tahmoor_west_report_6.pdf": f"{GM3}/2026/09/Report-6-–-25-March-2022-to-18-November-2022-for-Longwalls-West-1-to-West-4.pdf",
    "tahmoor_west_report_7.pdf": f"{GM3}/2026/09/Report-7-–-1-January-2022-to-31-December-2022-for-Longwalls-West-1-to-West-4.pdf",
    "tahmoor_west_report_8.pdf": f"{GM3}/2026/09/Report-8-–-1-January-2023-to-30-June-2023-for-Longwalls-West-1-to-West-4.pdf",
    # Tahmoor extraction plans (MSEC subsidence reports with the MGA-gridded drawings)
    "tahmoor_west_EP_app_A_subsidence_predictions.pdf": f"{TCOL}/11-app-a-subsidence-predictions.pdf",
    "tahmoor_west_EP_app_A_subsidence_predictions_1.pdf": f"{TCOL}/11-app-a-subsidence-predictions-1.pdf",
    "tahmoor_MSEC1192_south_EP_S1A-S6A_RevA.pdf":
        f"{GM3}/2026/10/MSEC1192-Tahmoor-South-Extraction-Plan-for-LW-S1A-S6A-Rev-A-Final-compressed_Redacted.pdf",
    "tahmoor_annual_review_2019.pdf": f"{GM3}/2026/09/Annual-Review-Report-–-Tahmoor-–-2019.pdf",
    # Appin (index https://gm3.au/appin-mine/)
    "appin_APNSVY00237_LW709-711_905_EP_variation_approved_plan_MGA2020.pdf":
        f"{GM3}/2025/02/APNSVY00237-LWs-709-711-and-LW905-Extraction-Plan-VARIATION-Approved-Plan-MGA2020-signed.pdf",
    "appin_APNSVY00238_LW709-711_905_EP_variation_plan2_surface_features_MGA2020.pdf":
        f"{GM3}/2025/02/APNSVY00238-LWs-709-711-and-LW905-Extraction-Plan-VARIATION-Plan-2-Surface-Features-MGA2020.pdf",
    "appin_APNSVY00250_LW709-711_905_EP_variation_plan3_geology_MGA2020.pdf":
        f"{GM3}/2025/02/APNSVY00250-LWs-709-711-and-LW905-Extraction-Plan-VARIATION-Plan-3-Geology-MGA2020.pdf",
    "appin_MSEC1050_LW902_EoP_report_RevA.pdf":
        f"{GM3}/2024/07/Attachment-B-MSEC1050-Appin-LW902-End-of-Panel-Report-Rev.-A.pdf",
    "appin_LW902_EoP_summary.pdf": f"{GM3}/2024/07/LW902_EOP_Summary_Report.pdf",
    "appin_area9_LW901-904_extraction_plan.pdf": f"{GM3}/2024/06/appin-area-9-longwalls-901-904-extraction-plan.pdf",
    "appin_LW903_EoP_monitoring_review.pdf":
        f"{GM3}/2024/07/Attachment-B-End-of-Panel-Subsidence-Monitoring-Review-Report-for-Appin-Longwall-903.pdf",
    "appin_LW904_EoP_monitoring_review.pdf": f"{GM3}/2024/07/Attachment-B-Appin-LW904-Subsidence-Monitoring-Review.pdf",
    "appin_LW905_EoP_monitoring_review.pdf": f"{GM3}/2024/07/Attachment-B-Appin-LW905-Subsidence-Monitoring-Review.pdf",
    "appin_LW901_MSEC_EoP_report.pdf": f"{GM3}/2024/07/Attachment-B-Longwall-901-MSEC-End-of-Panel-Report.pdf",
    "appin_LW707_monitoring_report.pdf": f"{GM3}/2024/07/Attachment-B-7-Longwall-707-Subsidence-Monitoring-Report.pdf",
    "appin_LW708_EoP_summary.pdf": f"{GM3}/2024/07/Appin-Longwall-708-End-of-Panel-Summary-Report.pdf",
    "appin_LW708_subsidence_assessment.pdf": f"{GM3}/2024/07/Attachment-D-LW708-Subsidence-Assessment.pdf",
    "appin_LW709_subsidence_assessment.pdf": f"{GM3}/2024/07/attachment-c-longwall-709-subsidence-assessment.pdf",
    "appin_LW904_EoP_summary.pdf": f"{GM3}/2024/07/Appin-LW904-End-of-Panel-Summary-Report.pdf",
    "appin_LW903_EoP_report.pdf": f"{GM3}/2024/07/Appin-Longwall-903-End-of-Panel-Report.pdf",
}

NOTES = {
    "tahmoor_west_EP_app_A_subsidence_predictions.pdf":
        "MSEC1019 Rev B (Jun 2019), Appendix A of the Tahmoor Western Domain LW W1-W2 Extraction Plan. Found by web "
        "search at the tahmoorcolliery.com.au URL; that URL answered 404 to a HEAD request on 8 Oct 2026 after the "
        "file was fetched, so the copy here is identified by its content (title page) and sha256.",
    "tahmoor_west_EP_app_A_subsidence_predictions_1.pdf":
        "MSEC1112 Rev A (Mar 2021), Appendix A of the Tahmoor Western Domain LW W3-W4 Extraction Plan "
        "(TAH-HSEC-326, Aug 2021 Ver2). Same remark as MSEC1019 about the URL.",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_size(url: str):
    try:
        r = subprocess.run(["curl", "-sIL", "--max-time", "30", url], capture_output=True, text=True, timeout=40)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    status, size = None, None
    for line in r.stdout.splitlines():
        low = line.lower()
        if low.startswith("http/"):
            status = line.split()[1]
        elif low.startswith("content-length:"):
            size = int(line.split(":")[1])
    return size, status


def build(check_urls: bool = True) -> dict:
    out = {}
    for name in sorted(p.name for p in SRC.glob("*.pdf")):
        p = SRC / name
        rec = {"url": SOURCES.get(name), "bytes": p.stat().st_size, "sha256": sha256(p),
               "retrieved": "2026-10-08"}
        if name in NOTES:
            rec["note"] = NOTES[name]
        if check_urls and rec["url"]:
            size, status = head_size(rec["url"])
            ok = status == "200" and size is not None
            rec["url_check_2026_10_08"] = {"http": status, "content_length": size if ok else None,
                                           "same_size": (size == rec["bytes"]) if ok else None}
        out[name] = rec
    return out


if __name__ == "__main__":
    res = build()
    (HERE.parent / "sources.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for k, v in res.items():
        print(k, v.get("url_check_2026_10_08"))

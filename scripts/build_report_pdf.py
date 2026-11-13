#!/usr/bin/env python3
"""Create the CRASH TEST PDF from verified experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "crash-test_research_report.pdf"


def main() -> None:
    results = json.loads((ROOT / "experiments" / "expanded" / "summary.json").read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=.7 * inch, rightMargin=.7 * inch, topMargin=.72 * inch, bottomMargin=.7 * inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("CrashTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=34, leading=36, textColor=colors.HexColor("#102017"), alignment=TA_LEFT, spaceAfter=12)
    subtitle = ParagraphStyle("CrashSub", parent=styles["Normal"], fontName="Helvetica", fontSize=13, leading=19, textColor=colors.HexColor("#45554a"), spaceAfter=22)
    heading = ParagraphStyle("CrashHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#102017"), spaceBefore=18, spaceAfter=7)
    body = ParagraphStyle("CrashBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.6, leading=14.5, textColor=colors.HexColor("#253128"), spaceAfter=10)
    small = ParagraphStyle("CrashSmall", parent=body, fontSize=8, leading=11, textColor=colors.HexColor("#5a685e"))
    model = results["held_out_2025"]["models"]["hist_gradient_boosting"]
    baseline = results["held_out_2025"]["models"]["seasonal_naive"]
    source = json.loads((ROOT / "data" / "provenance.json").read_text())["collision_source"]
    story = [Paragraph("CRASH TEST", title), Paragraph("Spatiotemporal Traffic Collision Risk Forecasting", subtitle), Paragraph("APPLIED URBAN SAFETY ANALYTICS / RESEARCH RELEASE 1.0", small), Spacer(1, 14)]
    sections = [
        ("01 / Research overview", f"CRASH TEST is a reproducible platform for weekly collision-risk forecasting across San Diego Police beats. The source snapshot contains {source['records']:,} unique reports across {source['unique_beats']} beats. It studies aggregate risk ranking, not individual crashes or exact crash locations."),
        ("02 / Problem definition", "The unit is a police beat and Monday-starting week. Primary target: next-week report count. Secondary target: next-week occurrence. The ranking objective identifies comparatively elevated aggregate risk, not causal danger or an operational allocation recommendation."),
        ("03 / Data sources and preprocessing", "The City of San Diego Police basic-report source supplies report time, police beat, injury, fatality, and hit-and-run fields. It is updated daily, excludes many property-damage-only events, and does not cover freeway crashes handled by CHP. Duplicate IDs and invalid dates/beats are removed; checksum, schema, coverage, and missingness are saved on each run."),
        ("04 / Feature engineering", "Lagged counts at 1, 2, 4, 8, 12, and 52 weeks, shifted trailing count and severity summaries, beat identity, and cyclical seasonality form the feature set. Every historical feature is shifted before the prediction week, preventing target leakage."),
        ("05 / Forecasting methodology", "Seasonal-naive and Poisson regression are compared with Poisson histogram gradient boosting. A distinct occurrence classifier is calibrated with logistic calibration. No deep-learning method is used because the available aggregate evidence does not justify its additional complexity."),
        ("06 / Model training and evaluation", "Training ends in 2023, validation is 2024, and 2025 is untouched test data. Two rolling origins evaluate 2023 and 2024 after their own training cutoffs. A deterministic disjoint police-beat test estimates transfer to unseen administrative areas."),
        ("07 / Results", f"On {results['held_out_2025']['rows']:,} held-out 2025 beat-weeks, gradient boosting achieved MAE {model['mae']:.3f}, compared with {results['held_out_2025']['models']['poisson_glm']['mae']:.3f} for Poisson regression and {baseline['mae']:.3f} for seasonal naive. Top-decile ranking lift was {model['lift']:.3f}."),
        ("08 / Prediction uncertainty", f"The occurrence classifier is calibrated on 2024 only and evaluated in 2025: average precision {results['occurrence_calibration']['average_precision']:.3f}, Brier score {results['occurrence_calibration']['brier_score']:.3f}. Bootstrap MAE improvement over seasonal naive is {results['bootstrap_mae_improvement_over_seasonal_naive']['mean_improvement']:.3f}; it does not guarantee behavior under distribution shift."),
        ("09 / Model interpretation", "Held-out permutation importance measures the count-error change when a feature is shuffled. It is predictive diagnostics, not causal attribution; correlated time-history variables can share or conceal apparent importance."),
        ("10 / Limitations", f"A disjoint evaluation of {results['spatial_generalization']['holdout_beats']} held-out beats yielded MAE {results['spatial_generalization']['hist_gradient_boosting']['mae']:.3f}. Reporting bias, property-damage-only omissions, CHP freeway coverage, administrative aggregation, absent verified point coordinates, uncertain exposure, and temporal/geographic shift limit generalization and operational use."),
        ("11 / Conclusions", "Leakage-safe historical features improve aggregate 2025 beat-week count forecasting relative to the evaluated seasonal baseline. The study does not demonstrate point-level prediction, causal drivers, or readiness for decisions about people or communities."),
        ("12 / Reproducibility", "Download official source CSVs, install dependencies, run the baseline and expanded scripts, then serve the read-only API and dashboard. The pipeline emits provenance checksums, processed panels, saved predictions, metrics, and figures. Raw and row-level data remain excluded from Git."),
    ]
    for index, (name, text) in enumerate(sections):
        story.extend([Paragraph(name, heading), Paragraph(text, body)])
        if index == 0:
            story.append(Image(str(ROOT / "reports" / "figures" / "collision_history.png"), width=6.85 * inch, height=2.85 * inch)); story.append(Spacer(1, 10))
        if index == 6:
            story.append(Image(str(ROOT / "reports" / "expanded-figures" / "model_comparison.png"), width=6.85 * inch, height=3.1 * inch)); story.append(Spacer(1, 10))
    story.extend([PageBreak(), Paragraph("References", heading), Paragraph("City of San Diego, Traffic collisions - basic reports: https://data.sandiego.gov/datasets/traffic-collisions/<br/>City of San Diego, Traffic collision details: https://data.sandiego.gov/datasets/traffic-collision-details/<br/>City of San Diego, Traffic ADT counts: https://data.sandiego.gov/datasets/traffic-adt-counts/", body)])
    doc.build(story)


if __name__ == "__main__":
    main()

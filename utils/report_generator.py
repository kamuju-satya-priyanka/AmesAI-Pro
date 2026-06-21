"""
report_generator.py
===================
PDF report generator using ReportLab (or fpdf2 as fallback).
Produces a professional toxicology report for a single compound.
"""
from __future__ import annotations

import io
import os
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


def _try_reportlab(profile: dict, smiles: str) -> bytes | None:
    """Attempt to generate PDF via reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=20, textColor=colors.HexColor("#00D4FF"),
            spaceAfter=6,
        )
        h1_style = ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=14, textColor=colors.HexColor("#E2E8F0"),
            spaceAfter=4, spaceBefore=12,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#94A3B8"),
            spaceAfter=4,
        )

        pred  = profile["prediction"]
        desc  = profile["descriptors"]
        lipi  = profile["lipinski"]
        pains = profile.get("pains", [])
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        story = []
        story.append(Paragraph("AmesAI Pro — Mutagenicity Report", title_style))
        story.append(Paragraph(f"Generated: {ts}", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B")))
        story.append(Spacer(1, 0.4*cm))

        # Prediction summary
        story.append(Paragraph("Prediction Summary", h1_style))
        pred_color = "#FF1744" if pred["class"] == 1 else "#00E676"
        summary_data = [
            ["Parameter", "Value"],
            ["SMILES", smiles[:60] + ("…" if len(smiles) > 60 else "")],
            ["Prediction", pred["label"]],
            ["Mutagenicity Probability", f"{pred['mut_prob']*100:.1f}%"],
            ["Confidence", f"{pred['confidence']:.1f}%"],
            ["Risk Category", pred["risk_label"]],
        ]
        t = Table(summary_data, colWidths=[5*cm, 11*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#00D4FF")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#0A0E1A")),
            ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#94A3B8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#0A0E1A"), colors.HexColor("#0D1220")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#1E293B")),
            ("PADDING",     (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Descriptors
        story.append(Paragraph("Molecular Descriptors", h1_style))
        desc_rows = [["Descriptor", "Value"]] + [[k, str(v)] for k, v in list(desc.items())[:12]]
        dt = Table(desc_rows, colWidths=[8*cm, 8*cm])
        dt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#7B2FBE")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#0A0E1A")),
            ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#94A3B8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#0A0E1A"), colors.HexColor("#0D1220")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#1E293B")),
            ("PADDING",     (0, 0), (-1, -1), 8),
        ]))
        story.append(dt)
        story.append(Spacer(1, 0.5*cm))

        # Lipinski
        story.append(Paragraph("Lipinski Rule of Five", h1_style))
        lipi_rows = [["Rule", "Value", "Limit", "Pass?"]]
        lipi_rows += [
            ["MW",   f"{lipi['MW']} Da",  "≤ 500 Da",  "✅" if lipi["MW"] <= 500  else "❌"],
            ["LogP", str(lipi["LogP"]),   "≤ 5",       "✅" if lipi["LogP"] <= 5   else "❌"],
            ["HBD",  str(lipi["HBD"]),    "≤ 5",       "✅" if lipi["HBD"] <= 5    else "❌"],
            ["HBA",  str(lipi["HBA"]),    "≤ 10",      "✅" if lipi["HBA"] <= 10   else "❌"],
        ]
        lt = Table(lipi_rows, colWidths=[4*cm, 4*cm, 4*cm, 4*cm])
        lt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#111827")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.HexColor("#00D4FF")),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#0A0E1A")),
            ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#94A3B8")),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#1E293B")),
            ("PADDING",     (0, 0), (-1, -1), 8),
        ]))
        story.append(lt)
        story.append(Spacer(1, 0.5*cm))

        # PAINS
        story.append(Paragraph("PAINS Alerts", h1_style))
        if pains:
            for p in pains:
                story.append(Paragraph(f"⚠️  {p}", body_style))
        else:
            story.append(Paragraph("✅  No PAINS alerts detected.", body_style))

        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E293B")))
        story.append(Paragraph(
            "Disclaimer: This report is generated for research purposes only "
            "and should not replace formal regulatory toxicology assessment.",
            body_style,
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None


def _try_fpdf(profile: dict, smiles: str) -> bytes | None:
    """Fallback PDF generation via fpdf2."""
    try:
        from fpdf import FPDF

        class PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 14)
                self.set_text_color(0, 212, 255)
                self.cell(0, 10, "AmesAI Pro — Mutagenicity Report", ln=True, align="C")
                self.set_draw_color(30, 41, 59)
                self.line(10, self.get_y(), 200, self.get_y())
                self.ln(4)

            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(100, 116, 139)
                self.cell(0, 10, f"Page {self.page_no()} | Research Use Only", align="C")

        pdf = PDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        pred  = profile["prediction"]
        desc  = profile["descriptors"]
        lipi  = profile["lipinski"]
        pains = profile.get("pains", [])
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 6, f"Generated: {ts}", ln=True)
        pdf.ln(4)

        def section(title: str, color=(0, 212, 255)):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*color)
            pdf.cell(0, 8, title, ln=True)
            pdf.set_draw_color(30, 41, 59)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(3)

        def row(label: str, value: str):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(226, 232, 240)
            pdf.cell(70, 7, label, ln=False)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(0, 7, str(value)[:80], ln=True)

        section("Prediction Summary")
        row("SMILES",    smiles[:55] + ("…" if len(smiles) > 55 else ""))
        row("Prediction", pred["label"])
        row("Mutagenicity Probability", f"{pred['mut_prob']*100:.1f}%")
        row("Confidence",  f"{pred['confidence']:.1f}%")
        row("Risk Category", pred["risk_label"])
        pdf.ln(4)

        section("Molecular Descriptors", color=(123, 47, 190))
        for k, v in list(desc.items())[:14]:
            row(k, str(v))
        pdf.ln(4)

        section("Lipinski Rule of Five", color=(0, 230, 118))
        row("MW",   f"{lipi['MW']} Da  {'(OK)' if lipi['MW']<=500 else '(FAIL)'}")
        row("LogP", f"{lipi['LogP']}  {'(OK)' if lipi['LogP']<=5 else '(FAIL)'}")
        row("HBD",  f"{lipi['HBD']}  {'(OK)' if lipi['HBD']<=5 else '(FAIL)'}")
        row("HBA",  f"{lipi['HBA']}  {'(OK)' if lipi['HBA']<=10 else '(FAIL)'}")
        row("Drug-likeable", "Yes" if lipi["DrugLikeable"] else f"No ({len(lipi['Violations'])} violations)")
        pdf.ln(4)

        section("PAINS Alerts", color=(255, 214, 0))
        if pains:
            for p in pains:
                row("⚠️", p)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(0, 230, 118)
            pdf.cell(0, 7, "No PAINS alerts detected.", ln=True)

        return bytes(pdf.output())

    except ImportError:
        return None


def generate_pdf_report(profile: dict, smiles: str) -> bytes | None:
    """
    Generate a PDF toxicology report.

    Tries ReportLab first, falls back to fpdf2.
    Returns PDF bytes or None if no PDF library is available.
    """
    pdf_bytes = _try_reportlab(profile, smiles)
    if pdf_bytes is not None:
        return pdf_bytes
    return _try_fpdf(profile, smiles)

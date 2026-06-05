#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "docs" / "research_metrics" / "research_output_metrics_report.md"
DEFAULT_OUT = ROOT / "docs" / "research_metrics" / "research_output_metrics_report.pdf"


def clean_inline(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("**", "")
        .replace("`", "")
        .replace(">", "")
    )


def build(src: Path = DEFAULT_SRC, out: Path = DEFAULT_OUT) -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=11.5,
        spaceAfter=5,
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0B7285"),
        spaceBefore=12,
        spaceAfter=7,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1F4D78"),
        spaceBefore=9,
        spaceAfter=5,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.4,
        leading=9.2,
    )

    story = []
    lines = src.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            story.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), h2))
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                parts = [p.strip() for p in tl.strip("|").split("|")]
                if all(set(p) <= {"-", ":"} for p in parts):
                    continue
                rows.append([Paragraph(clean_inline(p), small) for p in parts])
            if rows:
                width = 7.0 * inch
                col_widths = [width / len(rows[0])] * len(rows[0])
                tbl = Table(rows, colWidths=col_widths, repeatRows=1)
                tbl.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D5DC")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue
        elif line.startswith("- "):
            story.append(Paragraph("• " + clean_inline(line[2:]), body))
        elif line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ")):
            story.append(Paragraph(clean_inline(line), body))
        else:
            story.append(Paragraph(clean_inline(line), body))
        i += 1

    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Research Output Metrics Report",
    )
    doc.build(story)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_OUT
    print(build(src, out))

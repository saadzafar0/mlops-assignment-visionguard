"""Build the concise assignment PDF from Markdown and measured evidence."""

import argparse
import html
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import markdown
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTH = A4[0] - 88
NAVY = colors.HexColor("#14283F")
TEAL = colors.HexColor("#087F80")
GRAY = colors.HexColor("#536171")


def fonts():
    directory = Path("/usr/share/fonts/TTF")
    for name, file in [
        ("ReportSans", "DejaVuSans.ttf"),
        ("ReportBold", "DejaVuSans-Bold.ttf"),
        ("ReportItalic", "DejaVuSans-Oblique.ttf"),
        ("ReportMono", "DejaVuSansMono.ttf"),
    ]:
        pdfmetrics.registerFont(TTFont(name, str(directory / file)))
    pdfmetrics.registerFontFamily(
        "ReportSans",
        normal="ReportSans",
        bold="ReportBold",
        italic="ReportItalic",
        boldItalic="ReportBold",
    )


def styles():
    result = getSampleStyleSheet()
    definitions = {
        "Body": dict(fontName="ReportSans", fontSize=9.4, leading=13.2, spaceAfter=6),
        "H1": dict(
            fontName="ReportBold",
            fontSize=17,
            leading=22,
            textColor=NAVY,
            spaceBefore=1,
            spaceAfter=11,
            keepWithNext=True,
        ),
        "H2": dict(
            fontName="ReportBold",
            fontSize=11.2,
            leading=15,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "H3": dict(
            fontName="ReportBold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "Cell": dict(fontName="ReportSans", fontSize=8, leading=10.5, spaceAfter=0),
        "Caption": dict(
            fontName="ReportSans",
            fontSize=8,
            leading=10.5,
            textColor=GRAY,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "ReportCode": dict(
            fontName="ReportMono",
            fontSize=7.2,
            leading=10.2,
            spaceBefore=5,
            spaceAfter=7,
            backColor=colors.HexColor("#F0F4F7"),
            borderPadding=7,
        ),
        "CoverTitle": dict(
            fontName="ReportBold",
            fontSize=35,
            leading=43,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=12,
        ),
        "CoverSub": dict(
            fontName="ReportSans", fontSize=16, leading=23, textColor=TEAL, spaceAfter=20
        ),
    }
    for name, values in definitions.items():
        result.add(ParagraphStyle(name, alignment=TA_LEFT, **values))
    return result


def inline(element):
    text = html.escape(element.text or "")
    for child in element:
        contents = inline(child)
        if child.tag in {"strong", "b"}:
            text += "<b>" + contents + "</b>"
        elif child.tag in {"em", "i"}:
            text += "<i>" + contents + "</i>"
        elif child.tag == "code":
            text += '<font name="ReportMono" size="8.2">' + contents + "</font>"
        elif child.tag == "a":
            target = html.escape(child.get("href", ""), quote=True)
            text += f'<a href="{target}" color="#087F80">{contents}</a>'
        elif child.tag == "br":
            text += "<br/>"
        else:
            text += contents
        text += html.escape(child.tail or "")
    return text


def picture(path, width=WIDTH, maximum=195):
    with PILImage.open(path) as value:
        ratio = value.height / value.width
    height = width * ratio
    if height > maximum:
        width *= maximum / height
        height = maximum
    result = Image(str(path), width=width, height=height)
    result.hAlign = "CENTER"
    return result


def table(element, style):
    rows = []
    for row in element.findall(".//tr"):
        rows.append([Paragraph(inline(cell), style["Cell"]) for cell in row])
    count = len(rows[0])
    # Long model/candidate descriptions need more room than numeric columns.
    if count >= 6:
        widths = [WIDTH * 0.28] + [WIDTH * 0.72 / (count - 1)] * (count - 1)
    elif count == 5:
        widths = [WIDTH * 0.27] + [WIDTH * 0.73 / 4] * 4
    elif count == 3:
        widths = [WIDTH * 0.28, WIDTH * 0.35, WIDTH * 0.37]
    else:
        widths = [WIDTH / count] * count
    result = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DEEBEF")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, TEAL),
            ]
        )
    )
    return [result, Spacer(1, 7)]


def blocks(parent, style, figure_counter):
    story = []
    for element in parent:
        if element.tag in {"h1", "h2", "h3", "h4"}:
            story.append(
                Paragraph(inline(element), style[{"h1": "H1", "h2": "H2"}.get(element.tag, "H3")])
            )
        elif element.tag == "p":
            image = element.find("img")
            if image is not None:
                path = ROOT / "reports" / image.get("src")
                if path.exists():
                    figure_counter[0] += 1
                    story.append(
                        KeepTogether(
                            [
                                picture(path),
                                Paragraph(
                                    f"Figure {figure_counter[0]}. "
                                    + html.escape(image.get("alt", "")),
                                    style["Caption"],
                                ),
                            ]
                        )
                    )
                else:
                    story.append(
                        Paragraph(
                            "Screenshot unavailable: " + html.escape(image.get("alt", "")),
                            style["Caption"],
                        )
                    )
            else:
                story.append(Paragraph(inline(element), style["Body"]))
        elif element.tag in {"ul", "ol"}:
            for number, item in enumerate(element.findall("li"), 1):
                prefix = f"{number}. " if element.tag == "ol" else "• "
                story.append(Paragraph(prefix + inline(item), style["Body"]))
        elif element.tag == "table":
            story.extend(table(element, style))
        elif element.tag == "pre":
            value = "".join(element.itertext())
            story.append(XPreformatted(html.escape(value.rstrip()), style["ReportCode"]))
        elif element.tag == "hr":
            story.append(PageBreak())
        elif element.tag == "div" and element.get("class") == "gallery":
            images = element.findall("img")
            cells = []
            for image in images:
                path = ROOT / "reports" / image.get("src")
                if path.exists():
                    figure_counter[0] += 1
                    cells.append(
                        [
                            picture(path, WIDTH / len(images) - 9, 190),
                            Paragraph(
                                f"Figure {figure_counter[0]}. " + html.escape(image.get("alt", "")),
                                style["Caption"],
                            ),
                        ]
                    )
            if cells:
                layout = Table([cells], colWidths=[WIDTH / len(cells)] * len(cells))
                layout.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(layout)
        else:
            story.extend(blocks(element, style, figure_counter))
    return story


def page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E2E8"))
    canvas.line(44, A4[1] - 33, A4[0] - 44, A4[1] - 33)
    canvas.setFillColor(GRAY)
    canvas.setFont("ReportSans", 7.5)
    canvas.drawString(44, A4[1] - 25, "VISIONGUARD  |  23l-0829_A2_Ops")
    canvas.drawRightString(A4[0] - 44, A4[1] - 25, "MLOps Assignment 2")
    canvas.drawString(44, 25, "Saad Zafar · 23l-0829")
    canvas.drawRightString(A4[0] - 44, 25, str(document.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path, default=ROOT / "23l-0829_VisionGuard.pdf")
    args = parser.parse_args()
    revision = (
        args.source_revision
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    )
    fonts()
    style = styles()
    source = (ROOT / "reports/23l-0829_VisionGuard.md").read_text()
    source = source.replace("{{SOURCE_REVISION}}", revision).replace(
        "{{REPORT_DATE}}", datetime.now(ZoneInfo("Asia/Karachi")).strftime("%d %B %Y")
    )
    capture = ROOT / "reports/generated/report/curl_api_checks.json"
    if capture.exists():
        evidence = json.loads(capture.read_text())
        source = source.replace("{{CURL_CAPTURE_DATE}}", evidence["recorded_at_utc"])
    rendered = markdown.markdown(source, extensions=["tables", "fenced_code"])
    tree = ET.fromstring("<root>" + rendered + "</root>")
    story = blocks(tree, style, [0])
    document = SimpleDocTemplate(
        str(args.output),
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=48,
        bottomMargin=43,
        title="VisionGuard — MLOps Assignment 2",
        author="Saad Zafar (23l-0829)",
        subject="Data, models, tracking and cloud inference",
    )
    document.build(story, onFirstPage=page, onLaterPages=page)
    print(args.output)


if __name__ == "__main__":
    main()

"""Generate sample_docs/plans_and_support.pdf (two pages) so the demo includes a real PDF.

Requires the dev dependency `reportlab` (pip install -r requirements-dev.txt).
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "sample_docs" / "plans_and_support.pdf"

PAGES = [
    ("Lumora Cloud - Plans", [
        "Lumora Cloud offers three plans: Starter, Business and Enterprise.",
        "The Starter plan costs $12 per user per month and includes 50 GB of storage per workspace.",
        "The Business plan costs $29 per user per month, includes 1 TB of storage, and adds "
        "EU data residency and audit logs.",
        "The Enterprise plan is priced per contract and includes unlimited storage, single sign-on, "
        "and a dedicated customer success manager.",
        "All plans include a 14-day free trial. No credit card is needed to start a trial.",
    ]),
    ("Lumora Cloud - Support", [
        "Starter customers get email support with a first response within 2 business days.",
        "Business customers get email and chat support with a first response within 8 business hours.",
        "Enterprise customers get 24/7 phone support with a 1-hour first response for critical incidents.",
        "Support is available in English, German and Spanish.",
    ]),
]


def main():
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Title"]))
        for text in paragraphs:
            story += [Paragraph(text, styles["BodyText"]), Spacer(1, 10)]
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT), pagesize=A4).build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

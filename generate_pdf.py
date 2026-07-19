import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Investment Strategy Report 2026")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, "Confidential Document - Internal Use Only")
    text = [
        "1. Overview of Portfolio Management",
        "Diversifying asset allocation is a cornerstone of modern portfolio theory.",
        "We recommend allocating 60% of capital to low-cost equity index funds,",
        "30% to high-yield sovereign and corporate bonds, and 10% to liquid commodity assets.",
        "",
        "2. Asset Types and Target Growth",
        "Equity index funds (specifically tracking the S&P 500) target an annual growth rate of 8-10%.",
        "High-yield bonds provide a consistent coupon yield of 4.5% annually, hedging equity drawdowns.",
        "Commodity exposure (mainly physical gold ETFs) offers protection against high consumer inflation.",
        "",
        "3. Risk Assessment and Drawdown Mitigation",
        "Historical data shows that a 60/30/10 portfolio has a maximum draw-down of 12% during bear markets,",
        "compared to a 22% drawdown for an all-equity portfolio. Rebalancing should occur semi-annually."
    ]
    y = height - 110
    c.setFont("Helvetica", 11)
    for line in text:
        if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 11)
        else:
            c.drawString(50, y, line)
        y -= 20
    c.save()

if __name__ == "__main__":
    create_sample_pdf("documents/sample.pdf")

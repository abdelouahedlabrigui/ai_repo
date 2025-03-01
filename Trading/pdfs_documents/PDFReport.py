import base64
import datetime
from io import BytesIO
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from xml.sax.saxutils import escape


class PDFReport:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter)
        self.story = []
        self.styles = getSampleStyleSheet()
        self.setup_styles()

    def setup_styles(self):
        """Define custom styles for text formatting."""
        self.styles.add(ParagraphStyle(name="CustomNormal", fontSize=10, leading=14, spaceAfter=6))
        self.styles.add(ParagraphStyle(name="Header", fontSize=14, spaceAfter=10, leading=16, textColor=colors.darkblue))
        self.styles.add(ParagraphStyle(name="TableHeader", fontSize=12, textColor=colors.white, backColor=colors.black, alignment=1))
        self.styles.add(ParagraphStyle(name="Footer", fontSize=8, alignment=1, textColor=colors.grey))

    def format_interpretation(self, text):
        """Process interpretation text with formatting replacements."""
        # Convert markdown-style bold **bold** → <b>bold</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

        # Convert ### Header to <b>Header</b> (assumes single-line headers)
        text = re.sub(r'###\s*(.*?)\n', r'<b>\1</b><br/>', text)

        # Replace `\n\n` (double new lines) with `<br/><br/>` (double line break)
        text = text.replace("\n\n", "<br/><br/>")

        # Replace `\n` (single new line) with `<br/>`
        text = text.replace("\n", "<br/>")

        return text

    def add_header(self, title):
        """Add the document title as a header."""
        self.story.append(Paragraph(title, self.styles["Header"]))
        self.story.append(Spacer(1, 0.2 * inch))

    def add_table(self, data):
        """Create and style a summary table."""
        table_data = [
            ["Field", "Value"],
            ["Start Date", data["StartDate"]],
            ["End Date", data["EndDate"]],
            ["Ticker", data["Ticker"]],
            ["Created At", data["CreatedAT"]],
        ]
        table = Table(table_data, colWidths=[2.5 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.black),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2 * inch))

    def add_text(self, title, content, style="CustomNormal"):
        """Add formatted text content."""
        self.story.append(Paragraph(f"<b>{title}:</b>", self.styles["Header"]))
        self.story.append(Paragraph(content, self.styles[style]))
        self.story.append(Spacer(1, 0.2 * inch))

    def add_image(self, encoded_str):
        """Decode and add base64-encoded image."""
        try:
            image_data = base64.b64decode(encoded_str)
            image = Image(BytesIO(image_data), width=7 * inch, height=4 * inch)
            self.story.append(image)
            self.story.append(Spacer(1, 0.2 * inch))      
        except Exception as e:
            self.story.append(Paragraph(f"Error loading image: {str(e)}", self.styles["CustomNormal"]))

    def add_footer(self, canvas, doc):
        """Add page numbers in footer."""
        canvas.saveState()
        footer = Paragraph(f"Page {doc.page}", self.styles["Footer"])
        w, h = footer.wrap(doc.width, doc.bottomMargin)
        footer.drawOn(canvas, doc.leftMargin, h)
        canvas.restoreState()

    def build_pdf(self, data):
        """Compile all elements and generate the PDF."""
        self.add_header("Stock Analysis Report")
        self.add_table(data)
        # self.add_text("Paragraph", data["Pragraph"])
        self.add_text("Interpretation", self.format_interpretation(data["Interpretation"]))
        self.add_image(data["Visualization"])

        self.doc.build(self.story, onFirstPage=self.add_footer, onLaterPages=self.add_footer)
        return f"{self.filename}"


# Sample Data
# data = {
#     "Visualization": "base64_encoded_string_here",  # Replace with actual base64 string
#     "Ticker": "['AAPL', 'MSFT', 'JPM', 'XOM']",
#     "StartDate": "2024-01-01",
#     "EndDate": "2024-06-30",
#     "Pragraph": "This report provides an overview of stock volume trends using various predictive models.",
#     "Interpretation": "### Analysis Summary\n\nThe stock volumes indicate **potential surges** in the upcoming weeks.\n\n### Key Observations\n- AAPL shows a steady increase.\n- MSFT remains stable.",
#     "CreatedAT": str(datetime.datetime.now())
# }

# Generate PDF
# report = PDFReport("Stock_Report.pdf")
# report.build_pdf(data)

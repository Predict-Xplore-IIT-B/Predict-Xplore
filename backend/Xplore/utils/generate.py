import io
import os
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from django.conf import settings  # needed for accessing BASE_DIR

def generate_report(title, image, username) -> bytes:
    """
    Build the PDF in-memory and return bytes.
    `image` can be a file-like object or a path; we assume a file-like (BytesIO).
    """
    buffer = io.BytesIO()
    # Build a SimpleDocTemplate that writes into `buffer`
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=12, leftMargin=12,
                            topMargin=12, bottomMargin=6)
    elements = []
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(title, ParagraphStyle(name='Title',
                                                    fontName='Helvetica-Bold',
                                                    fontSize=28,
                                                    alignment=TA_CENTER)))
    elements.append(Spacer(1, 30))
    # Insert the image: ReportLab's Image flowable can take a file-like
    # Ensure `image` is at position 0
    try:
        image.seek(0)
    except Exception:
        pass
    elements.append(RLImage(image, 8*inch, 6*inch))
    # ... any other details you want to add ...
    # Build PDF into buffer
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# You can remove or ignore the old os.makedirs(...) and file writes.

    
def generate_complete_report():
    if not os.path.exists(os.path.join(settings.BASE_DIR, 'reports')):
        os.makedirs('reports')
import os
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from django.conf import settings


# currently reports are being saved in the file directory
def generate_report(title, image, username):
    if not os.path.exists(os.path.join(settings.BASE_DIR, 'reports')):
        os.makedirs('reports')

    document = []
    document.append(Spacer(1, 20))
    document.append(Paragraph(title, ParagraphStyle(name='Title',
                                                fontFamily='Georgia',
                                                fontSize=28,
                                                alignment=TA_CENTER)))

    document.append(Spacer(1, 30))
    # make the PDF responive while PDF generation, to be implemented later
    document.append(Image(image, 8*inch, 6*inch))
    # report details to be added

    # save in dir or return file
    SimpleDocTemplate(f'reports/{username}_{title}.pdf',pagesize=letter,
                    rightMargin=12, leftMargin=12,
                    topMargin=12, bottomMargin=6).build(document)
    
def generate_complete_report():
    if not os.path.exists(os.path.join(settings.BASE_DIR, 'reports')):
        os.makedirs('reports')
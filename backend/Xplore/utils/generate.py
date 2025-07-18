import os
import tempfile
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from django.conf import settings

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def generate_report(title, model_output_img, username, xai_img=None):
    logger.debug(f"Starting generate_report for username={username} title={title}")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from io import BytesIO

    # Ensure reports directory exists
    reports_dir = os.path.join(settings.BASE_DIR, 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        logger.debug(f"Created reports directory at {reports_dir}")
    
    logger.debug(f"Reports directory path: {reports_dir}")
    logger.debug(f"Reports directory exists: {os.path.exists(reports_dir)}")

    temp_files = []
    def save_img_to_temp(img):
        if img is None:
            logger.debug("No image provided to save_img_to_temp")
            return None
        if isinstance(img, BytesIO):
            img.seek(0)
            pil_img = PILImage.open(img).convert("RGB")
        elif isinstance(img, PILImage.Image):
            pil_img = img.convert("RGB")
        elif isinstance(img, (list, tuple)) or (hasattr(img, 'shape') and len(img.shape) >= 2):
            arr = img
            if hasattr(arr, "max") and arr.max() <= 1.0:
                arr = (arr * 255).astype('uint8')
            pil_img = PILImage.fromarray(arr).convert("RGB")
        else:
            raise ValueError("Unsupported image type for report generation.")
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp.close()
        temp_files.append(temp.name)
        logger.debug(f"Saved temporary image at {temp.name}")
        return temp.name

    model_img_path = save_img_to_temp(model_output_img)
    xai_img_path = save_img_to_temp(xai_img) if xai_img is not None else None

    document = []
    document.append(Paragraph("Model Output", ParagraphStyle(name='ModelTitle', fontSize=16, spaceAfter=10)))
    document.append(RLImage(model_img_path, 8*inch, 6*inch))
    if xai_img_path:
        document.append(Spacer(1, 20))
        document.append(Paragraph("XAI Explanation", ParagraphStyle(name='XAITitle', fontSize=16, spaceAfter=10)))
        document.append(RLImage(xai_img_path, 8*inch, 6*inch))

    filename = f'{username}_{title}.pdf'
    filepath = os.path.join(reports_dir, filename)
    logger.debug(f"Attempting to generate PDF at {filepath}")
    
    try:
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        doc.build(document)
        logger.debug(f"Successfully generated PDF at {filepath}")
        logger.debug(f"PDF file exists: {os.path.exists(filepath)}")
        logger.debug(f"PDF file size: {os.path.getsize(filepath)} bytes")
    except Exception as e:
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        raise

    # Clean up temp files
    for f in temp_files:
        try:
            os.remove(f)
            logger.debug(f"Cleaned up temp file {f}")
        except Exception as e:
            logger.error(f"Failed to clean up temp file {f}: {e}")

    return filename

def generate_complete_report():
    if not os.path.exists(os.path.join(settings.BASE_DIR, 'reports')):
        os.makedirs('reports')
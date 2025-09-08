# utils/generate.py

import os
import tempfile
import logging
from io import BytesIO
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from PIL import Image as PILImage
from django.utils import timezone

logger = logging.getLogger(__name__)

def _convert_to_pil(img_data):
    """A helper function to safely convert different image formats to a standard PIL Image."""
    if img_data is None:
        return None
    try:
        if isinstance(img_data, np.ndarray):
            # Convert numpy array (e.g., from OpenCV or a model mask) to PIL Image
            if img_data.max() <= 1.0: # Handle normalized masks
                img_data = (img_data * 255)
            arr = np.squeeze(img_data)
            if arr.ndim == 2:
                # This is a segmentation mask - convert to RGB using colormap
                arr = arr.astype(np.uint8)
                
                # Normalize the mask values to 0-1 range for colormap
                if arr.max() > 0:
                    # Scale class labels to use full colormap range
                    unique_classes = np.unique(arr)
                    if len(unique_classes) > 1:
                        # Map class indices to evenly distributed colormap values
                        arr_normalized = np.zeros_like(arr, dtype=np.float32)
                        for i, class_val in enumerate(unique_classes):
                            arr_normalized[arr == class_val] = i / (len(unique_classes) - 1)
                    else:
                        arr_normalized = arr.astype(np.float32)
                else:
                    arr_normalized = arr.astype(np.float32)
                
                import matplotlib.cm as cm
                # Use a more distinct colormap for segmentation
                arr_rgb = (cm.get_cmap('tab20')(arr_normalized)[:, :, :3] * 255).astype(np.uint8)
                return PILImage.fromarray(arr_rgb)
            if arr.ndim == 3 and arr.shape[2] in [1, 3]:
                arr = arr.astype(np.uint8)
                return PILImage.fromarray(arr)
            return PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")
        if isinstance(img_data, PILImage.Image):
            # Ensure image is in a standard RGB format
            return img_data.convert("RGB")
        
        logger.warning(f"Unsupported image type for report: {type(img_data)}")
        return None
    except Exception as e:
        logger.error(f"Failed to process image data for report: {e}")
        return None

def generate_report(title, model_output_img, username, xai_img=None):
    """
    Generates a PDF report in memory containing model outputs and optional XAI visualizations.
    """
    logger.debug(f"Generating report for {username} - {title}")
    
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    story = []
    temp_files = []

    try:
        # A list of images and their titles to add to the PDF
        images_to_add = [
            ("Model Output", model_output_img),
            ("XAI Explanation", xai_img)
        ]

        for img_title, img_data in images_to_add:
            if img_data is None:
                continue
            
            pil_img = _convert_to_pil(img_data)
            if not pil_img:
                continue

            # Create a temporary file to hold the image for the PDF library
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                pil_img.save(temp_file, format='PNG')
                temp_files.append(temp_file.name)
                
                # Add the title and image to the PDF story
                story.append(Paragraph(img_title, ParagraphStyle(name='Title', fontSize=16, spaceAfter=10)))
                story.append(RLImage(temp_file.name, width=6*inch, height=4.5*inch, kind='proportional'))
                story.append(Spacer(1, 0.25*inch))

        if not story:
            logger.error("No valid content could be generated for the PDF report.")
            return None, None

        # Build the PDF from the story
        doc.build(story)
        
        filename = f'{username}_{title}_{timezone.now().strftime("%Y%m%d%H%M%S")}.pdf'
        pdf_buffer.seek(0)
        return pdf_buffer, filename

    except Exception as e:
        logger.error(f"Failed to build the PDF document: {e}")
        return None, None
    finally:
        # CRITICAL: Ensure all temporary files are deleted, even if errors occurred.
        for f in temp_files:
            try:
                os.remove(f)
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {f}: {e}")


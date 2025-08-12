import os
import io
import json
import concurrent.futures
import logging
logger = logging.getLogger(__name__)
import numpy as np
import base64
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from rest_framework.response import Response
from rest_framework import status, permissions
from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse
from django.utils import timezone
from django.http import JsonResponse
from django.http import HttpResponse
from django.http import FileResponse
from django.contrib.auth import get_user_model
from django.conf import settings
from rest_framework.views import APIView
from .serializers import ModelOptionsSerializer
from .models import Model, Pipeline
from .serializers import ModelOptionsSerializer, ModelSerializer
from Architecture.architecture import load_image_segmentation
from utils.inference import image_segmentation
from utils.generate import generate_report
from predictor.models import Model
from utils.xai import generate_cam
import matplotlib
import tempfile
from django.core.files.base import ContentFile
from .models import TestCase, Report 
from django.utils import timezone
from utils.xai import generate_cam, generate_detection_cam
import shutil


# Configure matplotlib to use a non-interactive backend
matplotlib.use('Agg')
from Architecture.architecture import (
    load_image_segmentation,
    load_human_detection
)
from utils.inference import (
    image_segmentation,
    human_detection
)

# Get the user model
User = get_user_model()

class uploaded_image:
    session_test_image = None
    encoded_image = {}

def run_inference_call(weight, cv2_image):
    """
    Helper function to run inference based on model type.
    For HumanDetection, it returns the loaded model as well for XAI use.
    """
    try:
        if weight.model_type == 'ImageSegmentation':
            model, device = load_image_segmentation()
            model.load_state_dict(torch.load(weight.model_file, map_location=device))
            model.to(device)
            model.eval()
            # For segmentation, we return the mask output and the loaded model
            return image_segmentation(cv2_image, model, device), model

        if weight.model_type == 'HumanDetection':
            model = load_human_detection(weight.model_file.path)
            # For detection, return both the results object and the loaded model
            results = human_detection(cv2_image, model)
            return results, model 
            
    except Exception as e:
        logger.error(f"Error during inference for model {weight.name}: {e}")
    
    return None, None

class UploadModelView(APIView):
    def post(self, request):
        content_type = request.content_type

        if 'multipart/form-data' in content_type:
            name = request.data.get('name')
            description = request.data.get('description')
            model_file = request.data.get('model_file')
            created_by = request.data.get('created_by')
            model_type = request.data.get('model_type')
            model_thumbnail = request.data.get('model_image')  # <-- renamed

            if not all([name, description, model_file, created_by, model_type, model_thumbnail]):
                return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)

            new_model = Model.objects.create(
                name=name,
                description=description,
                model_file=model_file,
                created_by=User.objects.get(username=created_by),
                model_type=model_type,
                model_thumbnail=model_thumbnail,  # <-- renamed
                created_at=timezone.now()
            )

            new_model.save()
        else:
            return Response({'error': 'Please use form-data'}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

        return Response({'message': 'Model successfully uploaded.'}, status=status.HTTP_201_CREATED)

# views.py

# predictor/views.py

# --- Replace your old ImageUploadView with this one ---
class ImageUploadView(APIView):
    """
    Handles the initial image upload.
    Creates a TestCase in the database to reliably store the image for the next step.
    """
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "An image file is required."}, status=status.HTTP_400_BAD_REQUEST)

        test_case = TestCase.objects.create(
            created_by=request.user,
            test_image=image_file,
            status='Pending'
        )

        return Response({
            "message": "Image uploaded successfully and saved to a test case.",
            "test_case_id": test_case.id
        }, status=status.HTTP_201_CREATED)


# --- Replace your old PredictView with this one ---
# predictor/views.py

# --- REPLACE THIS ENTIRE CLASS ---
# predictor/views.py

# ... (keep all other imports and functions)

# predictor/views.py

# ... (keep all other imports and functions)

class PredictView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def post(self, request, *args, **kwargs):
        serializer = ModelOptionsSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data
        test_case_id = validated_data.get('test_case_id') 
        selected_models = validated_data.get("models", [])
        xai_algo = validated_data.get("xai_algo")
        target_class = validated_data.get("target_class")

        if not test_case_id:
            return Response({'error': 'A "test_case_id" is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            test_case = TestCase.objects.get(pk=test_case_id, created_by=request.user)
        except TestCase.DoesNotExist:
            return Response({'error': 'Test case not found.'}, status=status.HTTP_404_NOT_FOUND)

        try:
            image_path = test_case.test_image.path
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist for TestCase {test_case_id}: {image_path}")
                return Response({'error': f"Image file not found at path: {image_path}"}, status=status.HTTP_404_NOT_FOUND)
            cv2_image = cv2.imread(image_path)
            if cv2_image is None: raise ValueError("Failed to read image from path.")
        except Exception as e:
            logger.error(f"Error reading image for TestCase {test_case_id}: {e}")
            return Response({'error': 'Failed to process the uploaded image.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        weights = [Model.objects.get(name=name) for name in selected_models]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            inference_results = list(executor.map(run_inference_call, weights, [cv2_image]*len(weights)))

        report_urls = []
        for weight, result_tuple, model_name in zip(weights, inference_results, selected_models):
            if not result_tuple or result_tuple[0] is None:
                logger.warning(f"Skipping report for model '{model_name}': empty inference result.")
                continue

            try:
                xai_output_image, model_output_image = None, None
                model_output, loaded_model = result_tuple

                if weight.model_type == 'ImageSegmentation':
                    model_output_image = model_output[0] if isinstance(model_output, (list, tuple)) else model_output
                    
                    # Debug: Print shape and unique values
                    print(f"Segmentation mask shape: {model_output_image.shape}")
                    print(f"Unique values in mask: {np.unique(model_output_image)}")
                    print(f"Min: {model_output_image.min()}, Max: {model_output_image.max()}")
                    
                    # Ensure it's a proper segmentation mask
                    if model_output_image.ndim == 2:
                        # This is correct for segmentation masks
                        pass
                    elif model_output_image.ndim == 3 and model_output_image.shape[2] == 1:
                        model_output_image = model_output_image.squeeze(2)
                    
                    if xai_algo and target_class:
                        # --- Ensure classes are present and include 'forest' if missing ---
                        if not hasattr(weight, "classes") or not weight.classes or len(weight.classes) == 0:
                            weight.classes = ["forest"]
                        elif "forest" not in weight.classes:
                            weight.classes.append("forest")
                        # --- Check if target_class is in model's classes ---
                        if target_class not in weight.classes:
                            logger.error(f"Target category '{target_class}' not found in the list of classes for model '{model_name}'.")
                            return Response(
                                {'error': f"Target category '{target_class}' not found in the list of classes for model '{model_name}'. "
                                          f"Available classes: {weight.classes}"},
                                status=status.HTTP_400_BAD_REQUEST
                            )
                        rgb_image_for_xai = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) / 255.0

                        # --- Pad image so height and width are divisible by 32 ---
                        def pad_to_divisible(img, div=32, value=0):
                            h, w = img.shape[:2]
                            pad_h = (div - h % div) if h % div != 0 else 0
                            pad_w = (div - w % div) if w % div != 0 else 0
                            return np.pad(
                                img,
                                ((0, pad_h), (0, pad_w), (0, 0)),
                                mode='constant',
                                constant_values=value
                            )
                        rgb_image_for_xai_padded = pad_to_divisible(rgb_image_for_xai, 32, 0)

                        # Also pad/crop the mask to match the padded image shape
                        padded_h, padded_w = rgb_image_for_xai_padded.shape[:2]
                        mask_h, mask_w = model_output_image.shape[:2]
                        pad_h = padded_h - mask_h
                        pad_w = padded_w - mask_w
                        if pad_h >= 0 and pad_w >= 0:
                            model_output_image = np.pad(
                                model_output_image,
                                ((0, pad_h), (0, pad_w)),
                                mode='constant',
                                constant_values=0
                            )
                        else:
                            # If mask is larger, crop it to fit the padded image
                            model_output_image = model_output_image[:padded_h, :padded_w]

                        target_layers = [
                                loaded_model.decoder.blocks[2],
                                loaded_model.decoder.blocks[3],
                                loaded_model.decoder.blocks[4]
                        ]
                        xai_output_image = generate_cam(
                            model=loaded_model, rgb_img=rgb_image_for_xai_padded, model_output_mask=model_output_image,
                            target_layers=target_layers, target_category_name=target_class,
                            all_classes=weight.classes, algo=xai_algo,
                            device="cuda" if torch.cuda.is_available() else "cpu"
                        )

                elif weight.model_type == 'HumanDetection':
                    model_output_image = Image.fromarray(model_output.plot()[:, :, ::-1])
                    if xai_algo:
                        # --- FIX: Resize image to model's expected input size (e.g., 640x640) ---
                        # Ultralytics YOLO models typically expect 640x640 input
                        expected_size = (640, 640)
                        rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                        rgb_img_resized = cv2.resize(rgb_img, expected_size)
                        rgb_img_float = np.float32(rgb_img_resized) / 255.0
                        input_tensor = torch.from_numpy(np.transpose(rgb_img_float, (2, 0, 1))).unsqueeze(0)
                        input_tensor = input_tensor.requires_grad_()  # <-- Ensure requires_grad=True

                        # --- Wrap model for pytorch-grad-cam compatibility ---
                        class ModelWrapper(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model
                            def forward(self, x):
                                out = self.model(x)
                                # Return only the first element if tuple
                                return out[0] if isinstance(out, tuple) else out

                        cam_model = ModelWrapper(loaded_model.model)

                        target_layers = [
                            loaded_model.model.model[4],
                            loaded_model.model.model[6],
                            loaded_model.model.model[8]
                        ] 
                        
                        # --- Enable gradients for Grad-CAM ---
                        with torch.set_grad_enabled(True):
                            xai_output_image = generate_detection_cam(
                                model=cam_model, input_tensor=input_tensor,
                                target_layers=target_layers, rgb_img=rgb_img_float
                            )

                # --- Save model output image to model_output folder ---
                output_dir = os.path.join(settings.MEDIA_ROOT, "model_output")
                os.makedirs(output_dir, exist_ok=True)
                output_filename = f"{request.user.username}_{model_name}.png"
                output_path = os.path.join(output_dir, output_filename)
                img_to_save = None
                if isinstance(model_output_image, np.ndarray):
                    arr = np.squeeze(model_output_image)
                    # Always apply a color map for segmentation masks
                    if arr.ndim == 2:
                        arr = arr.astype(np.uint8)
                        import matplotlib.cm as cm
                        # Use a color map for all classes, not just binary
                        normed = arr / (arr.max() if arr.max() > 0 else 1)
                        arr_rgb = (cm.get_cmap('jet')(normed)[:, :, :3] * 255).astype(np.uint8)
                        img_to_save = Image.fromarray(arr_rgb)
                    elif arr.ndim == 3 and arr.shape[2] in [1, 3]:
                        arr = arr.astype(np.uint8)
                        img_to_save = Image.fromarray(arr)
                elif isinstance(model_output_image, Image.Image):
                    img_to_save = model_output_image
                if img_to_save:
                    img_to_save.save(output_path, format="PNG")

                pdf_buffer, report_filename = generate_report(
                    title=model_name, model_output_img=model_output_image,
                    username=request.user.username, xai_img=xai_output_image
                )

                if pdf_buffer:
                    report_instance = Report.objects.create(test_case=test_case)
                    report_instance.report_file.save(report_filename, ContentFile(pdf_buffer.read()), save=True)
                    download_url = request.build_absolute_uri(f'/model/download/report/{report_instance.id}')
                    report_urls.append({"model_name": model_name, "download_url": download_url})

            except Exception as e:
                logger.exception(f"Error processing model {model_name}: {e}")

        test_case.status = 'Completed'
        test_case.save()
        return Response({"message": "Inference and report generation complete.", "reports": report_urls}, status=status.HTTP_200_OK)

def home(request):
    return JsonResponse({"message": "Welcome to the Dashboard API"})

def model_list(request):
    models = Model.objects.values('id', 'name', 'description', 'model_type', 'created_at')
    return JsonResponse({"models": list(models)}, safe=False)

def report_list(request):
    reports = Report.objects.select_related('test_case__model').values(
        'id', 
        'test_case__id',
        'test_case__model__name',
        'test_case__model__model_type',
        'report_file', 
        'created_at'
    )
    return JsonResponse({"reports": list(reports)}, safe=False)

class CreateModelView(APIView):
    def post(self, request, *args, **kwargs):
        content_type = request.content_type

        if 'application/json' in content_type:
            data = request.data
            name = data.get('name')
            description = data.get('description')
            model_type = data.get('model_type')
            file_data = data.get('model_file')
            model_thumbnail = data.get('model_image')  # <-- renamed
            allowed_xai_models = data.get('allowed_xai_models')
            classes = data.get('classes')
            allowed_users = data.get('allowed_users')

            if not name or not description or not model_type or not file_data:
                return Response({"detail": "Required fields are missing."}, status=status.HTTP_400_BAD_REQUEST)

            if isinstance(file_data, str):
                file_content = base64.b64decode(file_data)

            if model_thumbnail and isinstance(model_thumbnail, str):
                model_thumbnail_content = base64.b64decode(model_thumbnail)
            else:
                model_thumbnail_content = None

            new_object = Model.objects.create(
                name=name,
                description=description,
                model_file=file_content,
                model_type=model_type,
                model_thumbnail=model_thumbnail_content,  # <-- renamed
                allowed_xai_models=allowed_xai_models or [],
                classes=classes or [],
                allowed_users=allowed_users or [],
                created_by=request.user
            )
            new_object.save()

        return Response({"detail": "Model created successfully."}, status=status.HTTP_201_CREATED)

def pipeline_list(request):
    pipelines = Pipeline.objects.values('id', 'name', 'is_active', 'created_at')
    return JsonResponse({"pipelines": list(pipelines)}, safe=False)

def create_pipeline(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            pipeline_name = data.get('name')
            is_active = data.get('is_active', True)

            if not pipeline_name:
                return JsonResponse({"message": "Missing required fields"}, status=400)

            pipeline_instance = Pipeline.objects.create(
                name=pipeline_name,
                is_active=is_active,
            )
            return JsonResponse({"message": "Pipeline created successfully", "pipeline_id": pipeline_instance.id}, status=201)

        except json.JSONDecodeError:
            return JsonResponse({"message": "Invalid JSON data"}, status=400)

    return JsonResponse({"message": "Only POST requests are allowed for this endpoint"}, status=405)

class FetchResultAPIView(APIView):
    def get(self, request, *args, **kwargs):
        username = request.query_params.get('username')
        model_name = request.query_params.get('model_name')

        if not username or not model_name:
            return JsonResponse({"error": "Missing required query parameters: 'username' and 'model_name'"}, status=400)

        model_key = f"{username}_{model_name}"

        if model_key in uploaded_image.encoded_image:
            return JsonResponse({
                "image": uploaded_image.encoded_image[model_key]
            }, status=200)

        return JsonResponse({"error": "Model results not found"}, status=404)

class FetchInferenceImage(APIView):
    # pass username and model-name in the URL
    def get(self, request, username, model_name, *args, **kwargs):
        # base64_image = request.session.get(f'{username}_{model_name}')
        base64_image = uploaded_image.encoded_image[f'{username}_{model_name}']
        if not base64_image:
            return HttpResponse("No image found in session.", status=400)

        img_data = base64.b64decode(base64_image)
        img_byte_arr = BytesIO(img_data)
        image = Image.open(img_byte_arr)

        processed_img_byte_arr = BytesIO()
        image.save(processed_img_byte_arr, format='PNG')
        processed_img_byte_arr.seek(0)

        response = HttpResponse(processed_img_byte_arr, content_type='image/png')
        response['Content-Disposition'] = 'inline; filename="inference_output.png"'

        return response

# views.py

class ReportDownloadView(APIView):
    permission_classes = [permissions.AllowAny]
    # permission_classes = [permissions.IsAuthenticated]
    print("ReportDownloadView called")
    def get(self, request, report_id):
        try:
            report = get_object_or_404(Report, pk=report_id)
            filename = report.report_file.name
            print(report.report_file)
            print(report_id)
            print("file downloading from", filename)

            
            # Check if the report file exists
            if not report.report_file:
                return Response({"error": "Report file not found."}, status=404)
            
            return FileResponse(
                report.report_file, 
                as_attachment=True, 
                filename=report.report_file.name
            )
        except Report.DoesNotExist:
            return Response({"error": "Report not found."}, status=404)
        except Exception as e:
            logger.error(f"Error downloading report {report_id}: {e}")
            return Response({"error": f"Error downloading report: {str(e)}"}, status=500)

from django.views import View
from django.http import FileResponse, Http404

class ModelOutputView(APIView):
    """
    GET /model/output/<username>/<model_name>
    Returns the model output image for the given user and model.
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, username, model_name, *args, **kwargs):
        output_dir = os.path.join(settings.MEDIA_ROOT, "model_output")
        output_filename = f"{username}_{model_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            return Response({"error": "Model output image not found."}, status=404)
        return FileResponse(open(output_path, "rb"), content_type="image/png")
import os
import io
import json
import concurrent.futures
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
import logging

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
    if weight.model_type == 'ImageSegmentation':
        # Load the segmentation model and device
        model, device = load_image_segmentation()

        # Load model weights
        model.load_state_dict(torch.load(weight.model_file, map_location=device))
        model.to(device)

        # Run inference
        return image_segmentation(cv2_image, model, device)

    if weight.model_type == 'HumanDetection':
        model = load_human_detection(weight.model_file.path)

        return human_detection(cv2_image, model)

class UploadModelView(APIView):
    def post(self, request):
        content_type = request.content_type

        if 'multipart/form-data' in content_type:
            name = request.data.get('name')
            description = request.data.get('description')
            model_file = request.data.get('model_file')
            created_by = request.data.get('created_by')
            model_type = request.data.get('model_type')
            model_image = request.data.get('model_image')

            if not all([name, description, model_file, created_by, model_type, model_image]):
                return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)

            new_model = Model.objects.create(
                name=name,
                description=description,
                model_file=model_file,
                created_by=User.objects.get(username=created_by),
                model_type=model_type,
                model_image=model_image,
                created_at=timezone.now()
            )

            new_model.save()
        else:
            return Response({'error': 'Please use form-data'}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

        return Response({'message': 'Model successfully uploaded.'}, status=status.HTTP_201_CREATED)

class ImageUploadView(APIView):
    def post(self, request, *args, **kwargs):
        content_type = request.content_type

        if 'multipart/form-data' in content_type:
            image_file = request.FILES.get('image')
            if not image_file:
                return HttpResponse("No image file uploaded.", status=400)

            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        elif 'application/json' in content_type:
            image_data = request.data.get('image')
            if not image_data:
                return HttpResponse("No base64 image data provided. Please provide the data.", status=400)

            base64_image = image_data
        elif 'image/jpeg' in content_type or 'image/png' in content_type:
            image_bytes = request.body
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
        else:
            return HttpResponse("Unsupported media type. Try uploading Jpeg format", status=415)

        uploaded_image.session_test_image = base64_image
        return HttpResponse("Image processed and stored in session.", status=200)

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.debug("Starting prediction request")
        
        username = request.data.get('username')
        serializer = ModelOptionsSerializer(data=request.data)

        if not User.objects.filter(username=username).exists():
            return Response({'error': 'User with this username does not exist. Please register the user.'}, status=404)

        base64_image = uploaded_image.session_test_image
        if not base64_image:
            return HttpResponse("No image found in session. Please upload the test image first.", status=400)

        img_data = base64.b64decode(base64_image)
        np_img = np.frombuffer(img_data, dtype=np.uint8)

        cv2_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if cv2_image is None:
            return HttpResponse("Failed to decode image.", status=400)

        if serializer.is_valid():
            selected_models = serializer.validated_data.get("models", [])
            xai_algo = serializer.validated_data.get("xai_algo", None)
            weights = []
            outputs = []
            report_filenames = []

            for model in selected_models:
                try:
                    weights.append(Model.objects.get(name=model))
                except Model.DoesNotExist:
                    return HttpResponse(f"Model '{model}' not found.", status=404)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda args: run_inference_call(*args), zip(weights, [cv2_image]*len(weights))))
                outputs = results

            for weight, report, model_name in zip(weights, outputs, selected_models):
                model_type = weight.model_type
                report_filename = f"{username}_{model_name}"
                logger.debug(f"Processing model output for {model_name}")
                
                try:
                    if model_type == 'ImageSegmentation':
                        msk = report[0] if isinstance(report, (list, tuple, np.ndarray)) and len(report) > 0 else report
                        # Save model output as base64 for API
                        import matplotlib.pyplot as plt
                        from io import BytesIO
                        buf = BytesIO()
                        plt.imshow(msk)
                        plt.axis("off")
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                        uploaded_image.encoded_image[f'{username}_{model_name}'] = encoded_image

                        # --- XAI logic ---
                        if xai_algo:
                            rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) / 255.0
                            mask_float = (msk == 1).astype(np.float32)
                            model_obj = weight
                            try:
                                target_layers = [
                                    model_obj.encoder.layer3,
                                    model_obj.encoder.layer4,
                                    model_obj.decoder.blocks[0].conv1[0],
                                    model_obj.decoder.blocks[1].conv1[0],
                                    model_obj.decoder.blocks[2].conv1[0],
                                    model_obj.decoder.blocks[3].conv1[0],
                                    model_obj.decoder.blocks[4].conv1[0],
                                ]
                            except Exception:
                                target_layers = [list(model_obj.children())[-1]]
                            cam_arr = generate_cam(
                                model=model_obj,
                                rgb_img=rgb_img,
                                mask=mask_float,
                                target_layers=target_layers,
                                algo=xai_algo,
                                device="cuda" if torch.cuda.is_available() else "cpu",
                                category=1
                            )
                            xai_img = cam_arr
                        # --- END XAI logic ---

                        # Generate and save report (with both images)
                        filename = generate_report(
                            model_name,
                            msk,
                            username,
                            xai_img=xai_img
                        )
                        report_path = os.path.join(settings.BASE_DIR, 'reports', filename)
                        logger.debug(f"Report path: {report_path}")
                        logger.debug(f"Report exists: {os.path.exists(report_path)}")
                        if os.path.exists(report_path):
                            logger.debug(f"Report file size: {os.path.getsize(report_path)} bytes")
                        
                        if not os.path.exists(report_path):
                            logger.error(f"PDF report was not generated at {report_path}")
                            logger.debug(f"Reports dir contents: {os.listdir(os.path.join(settings.BASE_DIR, 'reports'))}")
                            return Response({"error": "Report PDF was not generated."}, status=500)
                        # Return download URL for Insomnia
                        download_url = f"/model/download/report/{filename.replace('.pdf','')}"
                        report_filenames.append(download_url)

                    elif model_type == 'HumanDetection':
                        annotated_image = report.plot()
                        annotated_image = annotated_image[:, :, ::-1]
                        from PIL import Image as PILImage
                        image = PILImage.fromarray(annotated_image)
                        image_buffer = BytesIO()
                        image.save(image_buffer, format='png')
                        image_buffer.seek(0)
                        encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
                        uploaded_image.encoded_image[f'{username}_{model_name}'] = encoded_image

                        filename = generate_report(model_name, image, username)
                        report_path = os.path.join(settings.BASE_DIR, 'reports', filename)
                        logger.debug(f"Report path: {report_path}")
                        logger.debug(f"Report exists: {os.path.exists(report_path)}")
                        if os.path.exists(report_path):
                            logger.debug(f"Report file size: {os.path.getsize(report_path)} bytes")
                        
                        if not os.path.exists(report_path):
                            logger.error(f"PDF report was not generated at {report_path}")
                            logger.debug(f"Reports dir contents: {os.listdir(os.path.join(settings.BASE_DIR, 'reports'))}")
                            return Response({"error": "Report PDF was not generated."}, status=500)
                        download_url = f"/model/download/report/{filename.replace('.pdf','')}"
                        report_filenames.append(download_url)
                except Exception as e:
                    logger.exception(f"Error generating report for {model_name}: {e}")

            return Response({
                "message": "Inference Successful",
                "report_download_links": report_filenames
            }, status=200)

        return HttpResponse("Invalid data", status=400)

def home(request):
    return JsonResponse({"message": "Welcome to the Dashboard API"})

def model_list(request):
    models = Model.objects.values('id', 'name', 'description', 'model_type', 'created_at')
    return JsonResponse({"models": list(models)}, safe=False)

class CreateModelView(APIView):
    def post(self, request, *args, **kwargs):
        content_type = request.content_type

        if 'application/json' in content_type:
            data = request.data
            name = data.get('name')
            description = data.get('description')
            model_type = data.get('model_type')
            file_data = data.get('model_file')
            model_image = data.get('model_image')
            allowed_xai_models = data.get('allowed_xai_models')
            classes = data.get('classes')
            allowed_users = data.get('allowed_users')

            if not name or not description or not model_type or not file_data:
                return Response({"detail": "Required fields are missing."}, status=status.HTTP_400_BAD_REQUEST)

            if isinstance(file_data, str):
                file_content = base64.b64decode(file_data)

            if model_image and isinstance(model_image, str):
                model_image_content = base64.b64decode(model_image)
            else:
                model_image_content = None

            new_object = Model.objects.create(
                name=name,
                description=description,
                model_file=file_content,
                model_type=model_type,
                model_image=model_image_content,
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

class ReportDownloadView(APIView):
    def get(self, request, filename):
        report_path = os.path.join(settings.BASE_DIR, 'reports', f"{filename}.pdf")

        if os.path.exists(report_path):
            return FileResponse(open(report_path, 'rb'), content_type='application/pdf', filename=f'{filename}.pdf')
        return Response({'error': 'Report does not exist.'}, status=status.HTTP_404_NOT_FOUND)
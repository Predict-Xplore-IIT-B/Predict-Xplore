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
from django.contrib.auth import get_user_model
from django.conf import settings
from rest_framework.views import APIView
from .serializers import ModelOptionsSerializer, ModelSerializer
from .models import Model, Pipeline, Report, TestCase
from Architecture.architecture import load_image_segmentation
from Architecture.architecture import load_image_segmentation, load_human_detection
from utils.inference import image_segmentation, human_detection
from utils.generate import generate_report
import matplotlib
import uuid
from django.core.files.base import ContentFile
import logging
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

reports_dir = os.path.join(settings.MEDIA_ROOT, 'reports')
os.makedirs(reports_dir, exist_ok=True)

# Configure matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# Get the user model
User = get_user_model()

logger = logging.getLogger(__name__)

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
        username = request.data.get('username')
        serializer = ModelOptionsSerializer(data=request.data)

        if not User.objects.filter(username=username).exists():
            return Response({'error': 'User with this username does not exist.'}, status=status.HTTP_404_NOT_FOUND)

        base64_image = uploaded_image.session_test_image
        if not base64_image:
            return HttpResponse("No image found in session. Please upload the test image first.", status=400)

        try:
            img_data = base64.b64decode(base64_image)
            np_img = np.frombuffer(img_data, dtype=np.uint8)
            cv2_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        except Exception:
            return HttpResponse("Failed to decode image.", status=400)

        if cv2_image is None:
            return HttpResponse("Failed to decode image.", status=400)

        if serializer.is_valid():
            selected_models = serializer.validated_data.get("models", [])
            weights = []
            outputs = []

            for model_name in selected_models:
                if model_name == "DummyModel":
                    # Dummy output placeholder for DummyModel
                    outputs.append("DUMMY_OUTPUT")
                    weights.append(type("DummyWeight", (), {"model_type": "Dummy", "name": "DummyModel"})())
                else:
                    try:
                        weights.append(Model.objects.get(name=model_name))
                    except Model.DoesNotExist:
                        logger.error(f"Model '{model_name}' not found.")
                        return Response({"error": f"Model '{model_name}' not found in database. Please create/upload it first."}, status=404)

            if not weights:
                logger.error("No valid models found for prediction.")
                return Response({"error": "No valid models found for prediction."}, status=400)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_inference_call, weight, cv2_image) for weight in weights]
                for job in concurrent.futures.as_completed(futures):
                    result = job.result()
                    if result is not None:
                        outputs.append(result)
                    else:
                        logger.warning("Inference output is None for a model.")

            if not outputs:
                logger.error("Inference did not produce any outputs.")
                return Response({"error": "Inference did not produce any outputs."}, status=500)

            saved_reports = []
            test_case = None
            test_case_id = request.data.get('test_case_id')
            if test_case_id:
                try:
                    test_case = TestCase.objects.get(pk=test_case_id)
                except TestCase.DoesNotExist:
                    return Response({'error': 'TestCase not found.'}, status=status.HTTP_404_NOT_FOUND)

            for i, inference_output in enumerate(outputs):
                ml_weight = weights[i]
                model_name = ml_weight.name

                # DummyModel logic
                if model_name == "DummyModel":
                    buffer = BytesIO()
                    c = canvas.Canvas(buffer, pagesize=letter)
                    c.setFont("Helvetica-Bold", 20)
                    c.drawString(100, 700, "This is a dummy report for DummyModel.")
                    c.setFont("Helvetica", 12)
                    c.drawString(100, 680, "If you see this, your report saving and download logic works.")
                    c.save()
                    buffer.seek(0)
                    pdf_bytes = buffer.getvalue()
                    filename_pdf = f"dummy_{uuid.uuid4().hex}.pdf"
                    report_obj = Report()
                    if test_case:
                        report_obj.test_case = test_case
                    report_obj.report_file.save(filename_pdf, ContentFile(pdf_bytes), save=True)
                    report_obj.save()
                    saved_reports.append({
                        "model": model_name,
                        "report_id": report_obj.id,
                        "report_file_url": request.build_absolute_uri(report_obj.report_file.url),
                    })
                    continue

                # Ensure inference_output is always a list for segmentation, or wrap it
                if ml_weight.model_type == 'ImageSegmentation':
                    if not isinstance(inference_output, (list, tuple)):
                        inference_output = [inference_output]
                    for idx, msk in enumerate(inference_output):
                        image_stream = BytesIO()
                        plt.imshow(msk)
                        plt.axis("off")
                        plt.savefig(image_stream, format='png')
                        plt.close()
                        image_stream.seek(0)

                        try:
                            pdf_bytes = generate_report(model_name, image_stream, username)
                        except Exception as e:
                            logger.error(f"PDF generation failed: {str(e)}")
                            return Response({"error": f"PDF generation failed: {str(e)}"}, status=500)

                        filename_pdf = f"{username}_{model_name}_{uuid.uuid4().hex}.pdf"
                        report_obj = Report()
                        if test_case:
                            report_obj.test_case = test_case
                        try:
                            report_obj.report_file.save(filename_pdf, ContentFile(pdf_bytes), save=True)
                            report_obj.save()
                        except Exception as e:
                            logger.error(f"Failed to save report: {str(e)}")
                            return Response({"error": f"Failed to save report: {str(e)}"}, status=500)

                        if report_obj.report_file and report_obj.report_file.name:
                            saved_reports.append({
                                "model": model_name,
                                "report_id": report_obj.id,
                                "report_file_url": request.build_absolute_uri(report_obj.report_file.url),
                            })

                elif ml_weight.model_type == 'HumanDetection':
                    try:
                        annotated_image = inference_output.plot()
                        annotated_image = annotated_image[:, :, ::-1]
                        image = Image.fromarray(annotated_image)
                    except Exception as e:
                        logger.error(f"Failed to process output for {model_name}: {str(e)}")
                        return Response({"error": f"Failed to process output for {model_name}."}, status=500)

                    image_buffer = BytesIO()
                    image.save(image_buffer, format='png')
                    image_buffer.seek(0)

                    try:
                        pdf_bytes = generate_report(model_name, image_buffer, username)
                    except Exception as e:
                        logger.error(f"PDF generation failed: {str(e)}")
                        return Response({"error": f"PDF generation failed: {str(e)}"}, status=500)

                    filename_pdf = f"{username}_{model_name}_{uuid.uuid4().hex}.pdf"
                    report_obj = Report()
                    if test_case:
                        report_obj.test_case = test_case
                    try:
                        report_obj.report_file.save(filename_pdf, ContentFile(pdf_bytes), save=True)
                        report_obj.save()
                    except Exception as e:
                        logger.error(f"Failed to save report: {str(e)}")
                        return Response({"error": f"Failed to save report: {str(e)}"}, status=500)

                    if report_obj.report_file and report_obj.report_file.name:
                        saved_reports.append({
                            "model": model_name,
                            "report_id": report_obj.id,
                            "report_file_url": request.build_absolute_uri(report_obj.report_file.url),
                        })

            if not saved_reports:
                logger.error("No reports were generated or saved.")
                return Response({"error": "No reports were generated or saved."}, status=500)

            return Response({
                "message": "Inference successful, report(s) saved.",
                "reports": saved_reports
            }, status=status.HTTP_201_CREATED)

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
        model_key = f"{username}_{model_name}"
        base64_image = uploaded_image.encoded_image.get(model_key)
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
    def get(self, request, report_id):
        """
        Download a saved report by its Report ID.
        URL pattern should be: path('download/report/<int:report_id>/', ...)
        """
        try:
            report_obj = Report.objects.get(pk=report_id)
        except Report.DoesNotExist:
            return Response({'error': 'Report not found.'}, status=status.HTTP_404_NOT_FOUND)

        file_field = report_obj.report_file
        if not file_field or not file_field.name or not file_field.storage.exists(file_field.name):
            return Response({'error': 'No report file found.'}, status=status.HTTP_404_NOT_FOUND)

        response = FileResponse(file_field.open('rb'), content_type='application/pdf')
        download_name = os.path.basename(file_field.name)
        response['Content-Disposition'] = f'attachment; filename="{download_name}"'
        return response

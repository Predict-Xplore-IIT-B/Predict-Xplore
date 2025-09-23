# views.py
import os
import io
import json
import uuid
import zipfile
import shutil
import base64
import logging
import subprocess
import concurrent.futures
from datetime import datetime
from io import BytesIO

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from django.conf import settings
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.http import JsonResponse, HttpResponse, FileResponse

from rest_framework.views import APIView
from rest_framework import status, permissions
from rest_framework.response import Response
from .tasks import build_container_task

from .serializers import ModelOptionsSerializer, ModelSerializer
from .models import Model, Pipeline, Container, TestCase, Report

# Architecture / utils imports
from Architecture.architecture import (
    load_image_segmentation,
    load_human_detection
)
from utils.inference import (
    image_segmentation,
    human_detection
)
from utils.generate import generate_report
from utils.xai import generate_cam, generate_detection_cam

# Configure matplotlib to use a non-interactive backend
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

User = get_user_model()

class uploaded_image:
    session_test_image = None
    encoded_image = {}

def run_inference_call(weight, cv2_image):
    """
    Helper function to run inference based on model type.
    Returns (model_output, loaded_model) or (results, loaded_model)
    """
    try:
        if weight.model_type == 'ImageSegmentation':
            model, device = load_image_segmentation()
            # weight.model_file could be a FileField — handle accordingly
            model_path = getattr(weight, "model_file", None)
            if model_path:
                model_file_path = model_path.path if hasattr(model_path, "path") else model_path
                model.load_state_dict(torch.load(model_file_path, map_location=device))
            model.to(device)
            model.eval()
            return image_segmentation(cv2_image, model, device), model

        if weight.model_type == 'HumanDetection':
            # load_human_detection expects path
            model_file_path = getattr(weight, "model_file", None)
            model = load_human_detection(model_file_path.path if hasattr(model_file_path, "path") else model_file_path)
            results = human_detection(cv2_image, model)
            return results, model

    except Exception as e:
        logger.exception(f"Error during inference for model {getattr(weight, 'name', 'unknown')}: {e}")

    return None, None


class UploadModelView(APIView):
    def post(self, request):
        content_type = request.content_type or ""
        if 'multipart/form-data' in content_type:
            name = request.data.get('name')
            description = request.data.get('description')
            model_file = request.data.get('model_file')
            created_by = request.data.get('created_by')
            model_type = request.data.get('model_type')
            model_thumbnail = request.data.get('model_image')

            if not all([name, description, model_file, created_by, model_type, model_thumbnail]):
                return Response({'error': 'All fields are required.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                created_user = User.objects.get(username=created_by)
            except User.DoesNotExist:
                return Response({'error': 'Created_by user not found.'}, status=status.HTTP_404_NOT_FOUND)

            new_model = Model.objects.create(
                name=name,
                description=description,
                model_file=model_file,
                created_by=created_user,
                model_type=model_type,
                model_thumbnail=model_thumbnail,
                created_at=timezone.now()
            )
            new_model.save()
            return Response({'message': 'Model successfully uploaded.'}, status=status.HTTP_201_CREATED)
        else:
            return Response({'error': 'Please use form-data'}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)


class ImageUploadView(APIView):
    """
    Handles the initial image upload.
    Creates a TestCase to reliably store the image for the next step.
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


class PredictView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = ModelOptionsSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        test_case_id = validated_data.get('test_case_id')
        selected_models = validated_data.get("models", [])  # expecting model IDs
        xai_algo = validated_data.get("xai_algo")
        target_class = validated_data.get("target_class")

        if not test_case_id:
            return Response({'error': 'A "test_case_id" is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            test_case = TestCase.objects.get(pk=test_case_id, created_by=request.user)
        except TestCase.DoesNotExist:
            return Response({'error': 'Test case not found.'}, status=status.HTTP_404_NOT_FOUND)

        # Read image from filesystem via TestCase.test_image
        try:
            image_path = test_case.test_image.path
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist for TestCase {test_case_id}: {image_path}")
                return Response({'error': f"Image file not found at path: {image_path}"}, status=status.HTTP_404_NOT_FOUND)
            cv2_image = cv2.imread(image_path)
            if cv2_image is None:
                raise ValueError("Failed to read image from path.")
        except Exception as e:
            logger.exception(f"Error reading image for TestCase {test_case_id}: {e}")
            return Response({'error': 'Failed to process the uploaded image.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            weights = [Model.objects.get(pk=model_id) for model_id in selected_models]
        except Model.DoesNotExist as e:
            return Response({'error': f'One or more models not found: {e}'}, status=status.HTTP_404_NOT_FOUND)

        # Run inference in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            inference_results = list(executor.map(run_inference_call, weights, [cv2_image] * len(weights)))

        report_urls = []
        for weight, result_tuple, model_id in zip(weights, inference_results, selected_models):
            model_name = weight.name
            if not result_tuple or result_tuple[0] is None:
                logger.warning(f"Skipping report for model '{model_name}': empty inference result.")
                continue

            try:
                xai_output_image, model_output_image = None, None
                model_output, loaded_model = result_tuple

                if weight.model_type == 'ImageSegmentation':
                    # model_output might be (mask, other) or mask directly
                    model_output_image = model_output[0] if isinstance(model_output, (list, tuple)) else model_output

                    # Debugging info
                    logger.debug(f"Segmentation mask shape for {model_name}: {getattr(model_output_image, 'shape', 'unknown')}")
                    if isinstance(model_output_image, np.ndarray):
                        logger.debug(f"Segmentation mask unique values: {np.unique(model_output_image)}")

                    # Ensure mask dimensions
                    if isinstance(model_output_image, np.ndarray) and model_output_image.ndim == 3 and model_output_image.shape[2] == 1:
                        model_output_image = model_output_image.squeeze(2)

                    if xai_algo and target_class:
                        # Ensure classes exist
                        if not hasattr(weight, "classes") or not weight.classes or len(weight.classes) == 0:
                            weight.classes = ["forest"]
                        elif "forest" not in weight.classes:
                            weight.classes.append("forest")

                        if target_class not in weight.classes:
                            logger.error(f"Target category '{target_class}' not found in the list of classes for model '{model_name}'.")
                            return Response(
                                {'error': f"Target category '{target_class}' not found for model '{model_name}'. Available classes: {weight.classes}"},
                                status=status.HTTP_400_BAD_REQUEST
                            )

                        rgb_image_for_xai = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) / 255.0

                        # Pad to divisible by 32
                        def pad_to_divisible(img, div=32, value=0):
                            h, w = img.shape[:2]
                            pad_h = (div - h % div) if h % div != 0 else 0
                            pad_w = (div - w % div) if w % div != 0 else 0
                            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=value)

                        rgb_image_for_xai_padded = pad_to_divisible(rgb_image_for_xai, 32, 0)

                        # Pad/crop mask to match padded image
                        padded_h, padded_w = rgb_image_for_xai_padded.shape[:2]
                        mask_h, mask_w = model_output_image.shape[:2]
                        pad_h = padded_h - mask_h
                        pad_w = padded_w - mask_w
                        if pad_h >= 0 and pad_w >= 0:
                            model_output_image = np.pad(model_output_image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                        else:
                            model_output_image = model_output_image[:padded_h, :padded_w]

                        # NOTE: target_layers depend on your model architecture; adjust as required
                        try:
                            target_layers = [
                                loaded_model.decoder.blocks[2],
                                loaded_model.decoder.blocks[3],
                                loaded_model.decoder.blocks[4]
                            ]
                        except Exception:
                            # fallback if model structure different
                            target_layers = None

                        xai_output_image = generate_cam(
                            model=loaded_model,
                            rgb_img=rgb_image_for_xai_padded,
                            model_output_mask=model_output_image,
                            target_layers=target_layers,
                            target_category_name=target_class,
                            all_classes=weight.classes,
                            algo=xai_algo,
                            device="cuda" if torch.cuda.is_available() else "cpu"
                        )

                elif weight.model_type == 'HumanDetection':
                    # model_output expected to be ultralytics-like results
                    try:
                        # many detection libs have .plot()
                        model_output_image = Image.fromarray(model_output.plot()[:, :, ::-1])
                    except Exception:
                        # if model_output is already an array
                        if isinstance(model_output, np.ndarray):
                            model_output_image = Image.fromarray(model_output[:, :, ::-1])
                        else:
                            model_output_image = None

                    if xai_algo:
                        # Prepare input for detection cam
                        expected_size = (640, 640)
                        rgb_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                        rgb_img_resized = cv2.resize(rgb_img, expected_size)
                        rgb_img_float = np.float32(rgb_img_resized) / 255.0
                        input_tensor = torch.from_numpy(np.transpose(rgb_img_float, (2, 0, 1))).unsqueeze(0)
                        input_tensor = input_tensor.requires_grad_()

                        class ModelWrapper(torch.nn.Module):
                            def __init__(self, model):
                                super().__init__()
                                self.model = model
                            def forward(self, x):
                                out = self.model(x)
                                return out[0] if isinstance(out, tuple) else out

                        cam_model = ModelWrapper(loaded_model.model if hasattr(loaded_model, "model") else loaded_model)

                        # target layers selection may vary depending on model
                        try:
                            # attempt to grab conv layers in YOLO-like model
                            target_layers = [
                                loaded_model.model.model[4],
                                loaded_model.model.model[6],
                                loaded_model.model.model[8]
                            ]
                        except Exception:
                            target_layers = None

                        with torch.set_grad_enabled(True):
                            xai_output_image = generate_detection_cam(
                                model=cam_model,
                                input_tensor=input_tensor,
                                target_layers=target_layers,
                                rgb_img=rgb_img_float
                            )

                # Save model output image to model_output folder
                output_dir = os.path.join(settings.MEDIA_ROOT, "model_output")
                os.makedirs(output_dir, exist_ok=True)
                safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
                output_filename = f"{request.user.username}_{safe_name}.png"
                output_path = os.path.join(output_dir, output_filename)
                img_to_save = None

                if isinstance(model_output_image, np.ndarray):
                    arr = np.squeeze(model_output_image)
                    if arr.ndim == 2:
                        arr = arr.astype(np.uint8)
                        import matplotlib.cm as cm
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

                # Generate PDF report
                pdf_buffer, report_filename = generate_report(
                    title=model_name, model_output_img=model_output_image,
                    username=request.user.username, xai_img=xai_output_image
                )

                if pdf_buffer:
                    report_instance = Report.objects.create(test_case=test_case, model=weight)
                    report_instance.report_file.save(report_filename, ContentFile(pdf_buffer.read()), save=True)
                    download_url = request.build_absolute_uri(f'/model/download/report/{report_instance.id}')
                    report_urls.append({"model_name": model_name, "download_url": download_url})

            except Exception as e:
                logger.exception(f"Error processing model {model_name}: {e}")

        test_case.status = 'Completed'
        test_case.save()
        return Response({"message": "Inference and report generation complete.", "reports": report_urls}, status=status.HTTP_200_OK)


class PredictPipeline(APIView):
    """
    A simplified pipeline runner that reads base64 image stored in uploaded_image.session_test_image,
    runs multiple models sequentially and stores encoded outputs in uploaded_image.encoded_image.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        if not username or not User.objects.filter(username=username).exists():
            return Response({'error': 'User not found.'}, status=404)

        b64 = uploaded_image.session_test_image
        if not b64:
            return HttpResponse("Please upload first.", status=400)

        try:
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            temp_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if temp_image is None:
                raise ValueError("Failed to decode test image.")
        except Exception as e:
            logger.exception("Failed to decode base64 image for pipeline.")
            return HttpResponse("Failed to decode.", status=400)

        # Example pipeline model names; adjust as per your DB
        selected = ['HumanDetection', 'Segmentation', 'HumanDetection']
        try:
            weights = [Model.objects.get(name=m) for m in selected]
        except Model.DoesNotExist as e:
            logger.exception("Model not found in pipeline.")
            return HttpResponse("One or more selected models not found.", status=400)

        for name, weight in zip(selected, weights):
            result, loaded_model = run_inference_call(weight, temp_image)
            if result is None:
                logger.warning(f"Skipping {name} due to empty result.")
                continue

            # For segmentation we expect a mask -> convert to color
            if weight.model_type == 'ImageSegmentation':
                # result may be mask ndarray directly
                mask = result[0] if isinstance(result, (list, tuple)) else result
                out_img = self.mask_to_cv2(mask)
            else:
                # detection result assumed to have .plot() or be an ndarray
                try:
                    out_img = result.plot()
                except Exception:
                    out_img = result if isinstance(result, np.ndarray) else None

            if out_img is None:
                continue

            if hasattr(out_img, 'ndim') and out_img.ndim == 4:
                out_img = np.squeeze(out_img)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{name}_{ts}.png"
            save_dir = os.path.abspath(os.path.join("pipeline_outputs"))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, fname)
            try:
                # ensure BGR image when writing with cv2
                cv2.imwrite(save_path, out_img)
            except Exception:
                # fallback: convert with PIL
                try:
                    pil = Image.fromarray(out_img[:, :, ::-1])
                    pil.save(save_path)
                except Exception:
                    logger.exception("Failed to save pipeline output image.")

            # store base64-encoded output for fetching
            try:
                pil = Image.fromarray(out_img[:, :, ::-1])
                buf = BytesIO()
                pil.save(buf, format="PNG")
                buf.seek(0)
                uploaded_image.encoded_image[f"{username}_{name}"] = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                logger.exception("Failed to encode pipeline output.")

            # set temp_image to out_img for next step (if required)
            temp_image = out_img

        # Optionally display last image to matplotlib (non-interactive)
        try:
            plt.imshow(temp_image[:, :, ::-1])
            plt.axis('off')
            plt.close()
        except Exception:
            pass

        return HttpResponse("Pipeline inference successful", status=200)

    def mask_to_cv2(self, mask):
        mask = np.asarray(mask, dtype=np.int32)
        num_classes = int(mask.max() + 1) if mask.size else 1
        cmap = plt.get_cmap('tab20', num_classes)
        colors = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
        color_mask = colors[mask]
        # return BGR for cv2 usage
        return color_mask[..., ::-1]


def home(request):
    return JsonResponse({"message": "Welcome to the Dashboard API"})


def model_list(request):
    models = Model.objects.values(
        'id',
        'name',
        'description',
        'model_file',
        'created_by',
        'created_at',
        'model_type',
        'model_thumbnail',
        'allowed_xai_models',
        'classes',
    )
    return JsonResponse({"models": list(models)}, safe=False)


def report_list(request):
    qs = Report.objects.select_related('model', 'test_case').order_by('-created_at')
    data = []
    for r in qs:
        model_thumb_url = (request.build_absolute_uri(r.model.model_thumbnail.url) if (r.model and r.model.model_thumbnail) else None)
        data.append({
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "report_file": r.report_file.name if r.report_file else None,
            "report_file_url": request.build_absolute_uri(r.report_file.url) if r.report_file else None,
            "test_case__id": r.test_case_id,
            "model__id": r.model_id,
            "model__name": r.model.name if r.model else None,
            "model__model_type": r.model.model_type if r.model else None,
            "model__model_thumbnail_url": model_thumb_url,
        })
    return JsonResponse({"reports": data})


class CreateModelView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        content_type = request.content_type or ""
        if 'application/json' in content_type:
            data = request.data
            name = data.get('name')
            description = data.get('description')
            model_type = data.get('model_type')
            file_data = data.get('model_file')
            model_thumbnail = data.get('model_image')
            allowed_xai_models = data.get('allowed_xai_models')
            classes = data.get('classes')
            allowed_users = data.get('allowed_users')

            if not name or not description or not model_type or not file_data:
                return Response({"detail": "Required fields are missing."}, status=status.HTTP_400_BAD_REQUEST)

            # file_data expected as base64 string
            if isinstance(file_data, str):
                file_content = base64.b64decode(file_data)
            else:
                file_content = file_data

            model_thumbnail_content = None
            if model_thumbnail and isinstance(model_thumbnail, str):
                model_thumbnail_content = base64.b64decode(model_thumbnail)

            new_object = Model.objects.create(
                name=name,
                description=description,
                model_file=file_content,
                model_type=model_type,
                model_thumbnail=model_thumbnail_content,
                allowed_xai_models=allowed_xai_models or [],
                classes=classes or [],
                allowed_users=allowed_users or [],
                created_by=request.user
            )
            new_object.save()
            return Response({"detail": "Model created successfully."}, status=status.HTTP_201_CREATED)

        return Response({"detail": "Invalid content type."}, status=status.HTTP_400_BAD_REQUEST)


def pipeline_list(request):
    pipelines = Pipeline.objects.values('id', 'name', 'is_active', 'created_at')
    return JsonResponse({"pipelines": list(pipelines)}, safe=False)


def create_pipeline(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            pipeline_name = data.get('name')
            is_active = data.get('is_active', True)
            allowed_models = data.get('allowed_models', [])
            created_by_username = data.get('created_by')

            try:
                created_by = User.objects.get(username=created_by_username)
            except User.DoesNotExist:
                return JsonResponse({"message": "Created_by user not found"}, status=404)

            if not pipeline_name:
                return JsonResponse({"message": "Missing required fields"}, status=400)

            pipeline_instance = Pipeline.objects.create(
                name=pipeline_name,
                created_by=created_by,
                is_active=is_active,
                allowed_models=allowed_models
            )

            return JsonResponse({
                "message": "Pipeline created successfully",
                "pipeline_id": pipeline_instance.id
            }, status=201)

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
            return JsonResponse({"image": uploaded_image.encoded_image[model_key]}, status=200)

        return JsonResponse({"error": "Model results not found"}, status=404)


class FetchInferenceImage(APIView):
    # GET /model/output/<username>/<model_name>
    permission_classes = [permissions.AllowAny]

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
    permission_classes = [permissions.AllowAny]

    def get(self, request, report_id):
        try:
            report = get_object_or_404(Report, pk=report_id)
            if not report.report_file:
                return Response({"error": "Report file not found."}, status=404)

            return FileResponse(report.report_file, as_attachment=True, filename=os.path.basename(report.report_file.name))
        except Report.DoesNotExist:
            return Response({"error": "Report not found."}, status=404)
        except Exception as e:
            logger.exception(f"Error downloading report {report_id}: {e}")
            return Response({"error": f"Error downloading report: {str(e)}"}, status=500)


class ModelOutputView(APIView):
    """
    GET /model/output/<username>/<model_name>
    Returns the model output image saved in MEDIA_ROOT/model_output/<username>_<model_name>.png
    """
    permission_classes = [permissions.AllowAny]

    def get(self, request, username, model_name, *args, **kwargs):
        output_dir = os.path.join(settings.MEDIA_ROOT, "model_output")
        output_filename = f"{username}_{model_name}.png"
        output_path = os.path.join(output_dir, output_filename)
        if not os.path.exists(output_path):
            return Response({"error": "Model output image not found."}, status=404)
        return FileResponse(open(output_path, "rb"), content_type="image/png")


def container_list(request):
    container = Container.objects.values('id', 'name', 'description', 'created_at')
    return JsonResponse({"containers": list(container)}, safe=False)


class CreateContainer(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        name = request.data.get('name')
        description = request.data.get('description')
        allowed_users = request.data.get('allowed_users', [])
        zip_file = request.FILES.get("zipfile")

        if not all([name, description, zip_file]):
            return Response({"error": "Fields 'name', 'description', and 'zipfile' are required."}, status=status.HTTP_400_BAD_REQUEST)

        if Container.objects.filter(name=name).exists():
            return Response({"error": f"A container with the name '{name}' already exists."}, status=status.HTTP_400_BAD_REQUEST)

        # Read file content and encode it to pass to the task
        zip_file_content = zip_file.read()
        zip_file_content_b64 = base64.b64encode(zip_file_content).decode('utf-8')

        # Call the background task
        build_container_task.delay(
            name=name,
            description=description,
            allowed_users=allowed_users,
            created_by_id=request.user.id,
            zip_file_content_b64=zip_file_content_b64
        )
        
        return Response(
            {"message": f"Container '{name}' build process has been started in the background."},
            status=status.HTTP_202_ACCEPTED
        )
    
    def FileHandler(self, request, name):
        try:
            upload_dir = f"/app/uploads/{name}/"
            os.makedirs(upload_dir, exist_ok=True)

            if "zipfile" not in request.FILES:
                logger.error("Zip file not provided")
                return False

            zip_file = request.FILES["zipfile"]
            zip_path = os.path.join(upload_dir, f"{name}.zip")

            with open(zip_path, "wb+") as f:
                for chunk in zip_file.chunks():
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = [m for m in zip_ref.namelist() if not m.endswith("/")]
                root_folders = set(m.split("/")[0] for m in members)

                if len(root_folders) == 1:
                    root = list(root_folders)[0]
                    for member in members:
                        target_path = os.path.join(upload_dir, member[len(root) + 1:])
                        if member.endswith("/"):
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, "wb") as f:
                                f.write(zip_ref.read(member))
                else:
                    zip_ref.extractall(upload_dir)

            required_files = ["inference.py", "requirements.txt", "model.pth", "Dockerfile"]
            for rf in required_files:
                if not os.path.exists(os.path.join(upload_dir, rf)):
                    logger.error(f"Missing {rf} in uploaded zip")
                    return False

            return True
        except Exception as e:
            logger.exception(f"Exception in FileHandler: {e}")
            return False

    def clearDir(self, dir):
        if not os.path.exists(dir):
            return
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.exception(f'Failed to delete {file_path}. Reason: {e}')

    def buildContainer(self, name):
        image_name = f"user_{name}:latest"
        upload_dir = f"/app/uploads/{name}/"
        try:
            subprocess.run(["docker", "build", "-t", image_name, upload_dir], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.exception(f"Docker build failed: {e}")
            return False
        finally:
            # ensure cleaning up upload dir after build (optional)
            self.clearDir(upload_dir)


class RunContainer(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        image_name = request.data.get('image_name')

        job_id = str(uuid.uuid4())
        input_dir = os.path.abspath(f"./inputs/{job_id}/")
        output_dir = os.path.abspath(f"./outputs/{job_id}/")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        if 'test_file' not in request.FILES:
            return Response({'error': 'test_file is required'}, status=400)

        test_file = request.FILES['test_file'].name.replace(' ', '_')
        test_file_path = os.path.join(input_dir, test_file)

        with open(test_file_path, "wb+") as f:
            for chunk in request.FILES['test_file'].chunks():
                f.write(chunk)

        try:
            result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{input_dir}:/app/inputs",
                "-v", f"{output_dir}:/app/outputs",
                image_name,
                "python", "inference.py", f"/app/inputs/{test_file}"
            ], capture_output=True, text=True, check=True)

            logger.debug(f"Container output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.exception(f"Container failed: {e}\nstdout:{getattr(e, 'stdout', '')}\nstderr:{getattr(e, 'stderr', '')}")
            return Response({'error': 'Container execution failed', 'stdout': getattr(e, 'stdout', ''), 'stderr': getattr(e, 'stderr', '')}, status=500)

        results_csv = os.path.join(output_dir, "results.csv")
        if os.path.exists(results_csv):
            return Response({'detail': 'Inference complete', 'results_file': results_csv}, status=200)
        else:
            return Response({'error': 'No results generated'}, status=500)

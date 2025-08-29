import csv
from datetime import datetime
import os
import io
import json
import concurrent.futures
import shutil
import subprocess
import uuid
import zipfile
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
from .models import Model, Pipeline, Container
from .serializers import ModelOptionsSerializer, ModelSerializer
from Architecture.architecture import load_image_segmentation
from utils.inference import image_segmentation
from utils.generate import generate_report
from predictor.models import Model
import matplotlib

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
            image_stream = BytesIO()
            weights = []
            outputs = []

            for model in selected_models:
                try:
                    weights.append(Model.objects.get(name=model))
                except Model.DoesNotExist:
                    return HttpResponse(f"Model '{model}' not found.", status=404)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(run_inference_call, weight, cv2_image) for weight in weights]
                for job in concurrent.futures.as_completed(futures):
                    outputs.append(job.result())

            for i, report in enumerate(outputs):
                if weights[i].model_type == 'ImageSegmentation':
                    for msk in report:
                        plt.imshow(msk)
                        plt.axis("off")
                        plt.savefig(image_stream, format='png')
                        image_stream.seek(0)
                        encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
                        # request.session[f'{username}_{selected_models[i]}'] = encoded_image
                        uploaded_image.encoded_image[f'{username}_{selected_models[i]}'] = encoded_image

                        generate_report(selected_models[i], image_stream, username)

                if weights[i].model_type == 'HumanDetection':
                    annotated_image = report.plot()
                    annotated_image = annotated_image[:, :, ::-1]
                    image = Image.fromarray(annotated_image)
                    image_buffer = io.BytesIO()
                    image.save(image_buffer, format='png')
                    image_buffer.seek(0)
                    encoded_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
                    uploaded_image.encoded_image[f'{username}_{selected_models[i]}'] = encoded_image

                    generate_report(selected_models[i], image_buffer, username)

            return HttpResponse("Inference Successful", status=200)

        return HttpResponse("Invalid data", status=400)
    
class PredictPipeline(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        if not User.objects.filter(username=username).exists():
            return Response({'error': 'User not found.'}, status=404)

        
        b64 = uploaded_image.session_test_image
        if not b64:
            return HttpResponse("Please upload first.", status=400)
        data = base64.b64decode(b64)
        arr  = np.frombuffer(data, dtype=np.uint8)
        temp_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if temp_image is None:
            return HttpResponse("Failed to decode.", status=400)

        selected = ['HumanDetection','Segmentation','HumanDetection']  
        weights  = [Model.objects.get(name=m) for m in selected]

        for name, weight in zip(selected, weights):
            pred = run_inference_call(weight, temp_image)

            
            if weight.model_type == 'ImageSegmentation':
                out_img = self.mask_to_cv2(pred)
            else:
                out_img = pred.plot()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{name}_{ts}.png"
            save_path = os.path.join(
                "E:/External_Projects/Predict-Xplore/PipelineOutputs",
                fname
            )
            if out_img.ndim == 4:
                out_img = np.squeeze(out_img)
            cv2.imwrite(save_path, out_img)

            
            pil = Image.fromarray(out_img[:, :, ::-1])  
            buf = BytesIO()
            pil.save(buf, format="PNG")
            buf.seek(0)
            uploaded_image.encoded_image[f"{username}_{name}"] = base64.b64encode(buf.getvalue()).decode()

            
            temp_image = out_img
        plt.imshow(temp_image)

        return HttpResponse("Pipeline inference successful", status=200)

    
    def mask_to_cv2(self, mask):
        
        num_classes=mask.max() + 1
        colors = cmap = plt.get_cmap('tab20', num_classes)  # or 'nipy_spectral', 'gist_ncar'
        colors = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8) 
        color_mask = colors[mask]
        return color_mask[..., ::-1] 


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
            allowed_models = data.get('allowed_models', [])
            created_by_username = data.get('created_by')

            created_by = User.objects.get(username=created_by_username)

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

class CreateContainer(APIView):
    def post(self, request, *args, **kwargs):

        data = request.data
        name = data.get('name')
        description = data.get('description')
        allowed_users = data.get('allowed_users', [])
        upload_dir = f"/app/uploads/{name}/"

        if not self.FileHandler(request, name):
            return JsonResponse({"error": "Error in folder processing"}, status=400)

        if not self.buildContainer(name):
            self.clearDir(upload_dir)
            return JsonResponse({"error": "Error in Building Container"}, status=400)
        
        user = request.user if request.user.is_authenticated else User.objects.first()

        container = Container.objects.create(
            name=name,
            description=description,
            allowed_users=allowed_users,
            created_by=user
        )
        return Response({"detail": "Container created successfully."}, status=status.HTTP_201_CREATED)

    def FileHandler(self, request, name):
        try:
            upload_dir = f"/app/uploads/{name}/"
            os.makedirs(upload_dir, exist_ok=True)

            if "zipfile" not in request.FILES:
                print("Zip file not provided")
                return False

            zip_file = request.FILES["zipfile"]
            zip_path = os.path.join(upload_dir, f"{name}.zip")

            # Save the uploaded zip
            with open(zip_path, "wb+") as f:
                for chunk in zip_file.chunks():
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                
                members = [m for m in zip_ref.namelist() if not m.endswith("/")]
                root_folders = set(m.split("/")[0] for m in members)

                if len(root_folders) == 1:
                    
                    root = list(root_folders)[0]
                    for member in members:

                        target_path = os.path.join(upload_dir, member[len(root)+1:])
                        if member.endswith("/"):
                            os.makedirs(target_path, exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with open(target_path, "wb") as f:
                                f.write(zip_ref.read(member))
                else:
                    
                    zip_ref.extractall(upload_dir)
            
            
            print("Extracted files:", os.listdir(upload_dir))


            required_files = ["inference.py", "requirements.txt", "model.pth", "Dockerfile"]
            for rf in required_files:
                if not os.path.exists(os.path.join(upload_dir, rf)):
                    print(f"Missing {rf} in uploaded zip")
                    return False

            return True
        except Exception as e:
            print(f"Exception in FileHandler: {e}")
            return False
        
    def clearDir(self, dir):
        for filename in os.listdir(dir):
                file_path = os.path.join(dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

    def buildContainer(self, name):
        image_name = f"user_{name}:latest"
        upload_dir = f"/app/uploads/{name}/"
        try:
            subprocess.run(["docker", "build", "-t", image_name, upload_dir], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Docker build failed: {e}")
            return False
        finally:
            self.clearDir(upload_dir)


class RunContainer(APIView):
    def post(self, request, *args, **kwargs):
        username = request.data.get('username')
        image_name = request.data.get('image_name')

        job_id = str(uuid.uuid4())
        
        # Use absolute paths that exist on host
        input_dir = os.path.abspath(f"./inputs/{job_id}/")  
        output_dir = os.path.abspath(f"./outputs/{job_id}/")
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Save the video file
        video_filename = request.FILES['video'].name.replace(' ', '_')
        video_path = os.path.join(input_dir, video_filename)
        
        with open(video_path, "wb+") as f:
            for chunk in request.FILES['video'].chunks():
                f.write(chunk)

        print(f"Video saved to: {video_path}")
        print(f"File size: {os.path.getsize(video_path)} bytes")
        print(f"Directory contents: {os.listdir(input_dir)}")

        # Run container
        try:
            result = subprocess.run([
                "docker", "run", "--rm",
                "-v", f"{input_dir}:/app/inputs",
                "-v", f"{output_dir}:/app/outputs",
                image_name,
                "python", "inference.py", f"/app/inputs/{video_filename}"
            ], capture_output=True, text=True, check=True)
            
            print("Container output:", result.stdout)
            
        except subprocess.CalledProcessError as e:
            print(f"Container failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return Response({'error': 'Container execution failed'}, status=500)

        # Check for results
        results_csv = os.path.join(output_dir, "results.csv")
        if os.path.exists(results_csv):
            return Response({'detail': 'Inference complete', 'results_file': results_csv}, status=200)
        else:
            return Response({'error': 'No results generated'}, status=500)

            
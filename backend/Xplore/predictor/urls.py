from django.urls import path
from .views import (
    ImageUploadView,
    PredictView,
    model_list,
    pipeline_list,
    create_pipeline,
    home,
    FetchInferenceImage,
    ReportDownloadView,
    CreateModelView,
    UploadModelView,
    PredictPipeline,
    CreateContainer,
    RunContainer
)

urlpatterns = [
    path('instance/upload', ImageUploadView.as_view(), name='image-upload'),
    path('instance/predict', PredictView.as_view(), name='predict-instance'),
    path('list/', model_list, name='model-list'),
    path('pipelines/', pipeline_list, name='pipeline-list'),
    path('pipelines/predict', PredictPipeline.as_view(), name='predict-pipeline'),
    path('pipelines/create/', create_pipeline, name='pipeline-create'),
    path('home/', home, name='home'),  # Home page for the API
    path('output/<str:username>/<str:model_name>', FetchInferenceImage.as_view(), name='fetch-inference-image'),
    path('download/report/<str:filename>', ReportDownloadView.as_view(), name='download-report'),
    path('create-model/', CreateModelView.as_view(), name='create-model'),  # For creating models
    path('home/', home, name='home'),  # Use the `home` function
    path('output/<str:username>/<str:model_name>', FetchInferenceImage.as_view()),
    path('download/report/<str:filename>', ReportDownloadView.as_view()),
    path('create',UploadModelView.as_view()),
    path('create-container/', CreateContainer.as_view(), name='create-container'),
    path('run-container/', RunContainer.as_view(), name='run-container'),
]

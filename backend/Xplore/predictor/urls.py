from django.urls import path, re_path
from .views import (
    ImageUploadView,
    PredictView,
    model_list,
    report_list,
    pipeline_list,
    create_pipeline,
    home,
    container_list,
    FetchInferenceImage,
    ReportDownloadView,
    CreateModelView,
    UploadModelView,
    ModelOutputView,
    PredictPipeline,
    CreateContainer,
    RunContainer,
    stream_video
)

urlpatterns = [
    path('instance/upload', ImageUploadView.as_view(), name='image-upload'),
    path('instance/predict', PredictView.as_view(), name='predict-instance'),
    path('list/', model_list, name='model-list'),
    path('report/', report_list, name='report-list'),
    path('pipelines/', pipeline_list, name='pipeline-list'),
    path('pipelines/predict', PredictPipeline.as_view(), name='predict-pipeline'),
    path('pipelines/create/', create_pipeline, name='pipeline-create'),
    path('home/', home, name='home'),  # Home page for the API
    path('output/<str:username>/<str:model_name>/', ModelOutputView.as_view(), name='model-output'),
    path('download/report/<int:report_id>/', ReportDownloadView.as_view(), name='download-report'),
    path('create-model/', CreateModelView.as_view(), name='create-model'),  # For creating models
    path('home/', home, name='home'),  # Use the `home` function
    path('output/<str:username>/<str:model_name>', FetchInferenceImage.as_view()),
    path('download/report/<str:filename>', ReportDownloadView.as_view()),
    path('create',UploadModelView.as_view()),
    path('create-container/', CreateContainer.as_view(), name='create-container'),
    path('run-container/', RunContainer.as_view(), name='run-container'),
    path('list-container/', container_list, name='list-container'),
    path('outputs/<uuid:job_id>/<str:filename>', stream_video, name='stream_video'),
    path('download/report/<str:report_id>', ReportDownloadView.as_view()),
]

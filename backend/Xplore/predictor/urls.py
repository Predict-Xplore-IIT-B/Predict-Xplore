from django.urls import path
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
    PredictPipeline,
    CreateContainer,
    RunContainer,
    ModelOutputView,
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

    # Model outputs & reports
    path('output/<str:username>/<str:model_name>/', ModelOutputView.as_view(), name='model-output'),
    path('output/<str:username>/<str:model_name>', FetchInferenceImage.as_view()),  # without trailing slash
    path('download/report/<int:report_id>/', ReportDownloadView.as_view(), name='download-report'),
    path('download/report/<str:filename>', ReportDownloadView.as_view()),  # alternate download by filename

    # Model creation & upload
    path('create-model/', CreateModelView.as_view(), name='create-model'),
    path('create', UploadModelView.as_view()),

    # Container operations
    path('create-container/', CreateContainer.as_view(), name='create-container'),
    path('run-container/', RunContainer.as_view(), name='run-container'),
    path('list-container/', container_list, name='list-container'),
]

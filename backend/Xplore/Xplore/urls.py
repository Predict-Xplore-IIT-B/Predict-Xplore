# Xplore/urls.py

from django.contrib import admin
from django.urls import path, include

# --- ADD THESE TWO IMPORTS ---
# These are necessary to serve user-uploaded files during development.
from django.conf import settings
from django.conf.urls.static import static

from predictor.views import stream_video

urlpatterns = [
    path('admin/', admin.site.urls),

    # All auth & user-management routes live under /auth/
    path('auth/', include('users.urls')),

    # Predictor app (inference, models, etc.)
    path('model/', include('predictor.urls')),

    # Remove this line:
    # path('model/output/<str:username>/<str:model_name>/', ModelOutputView.as_view(), name='model-output'),
]

# --- ADD THIS SNIPPET AT THE END OF THE FILE ---
# This line tells Django's development server how to find and serve the images
# you upload. It is the standard and required way to handle media files.
if settings.DEBUG:
    import mimetypes
    mimetypes.add_type("video/mp4", ".mp4", True)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += [
        path('outputs/<uuid:job_id>/<str:filename>', stream_video, name='stream_video'),
    ]
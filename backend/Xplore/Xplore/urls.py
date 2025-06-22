from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from predictor.views import ImageUploadView

urlpatterns = [
    path('admin/', admin.site.urls),

    # All auth & user-management routes live under /auth/
    path('auth/', include('users.urls')),

    # Predictor app (inference, models, etc.)
    path('model/', include('predictor.urls')),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
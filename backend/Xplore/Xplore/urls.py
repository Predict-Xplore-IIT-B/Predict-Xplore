from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # All auth & user-management routes live under /auth/
    path('auth/', include('users.urls')),

    # Predictor app (inference, models, etc.)
    path('model/', include('predictor.urls')),
]

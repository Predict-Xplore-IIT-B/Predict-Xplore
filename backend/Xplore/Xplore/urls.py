from django.contrib import admin
from django.urls import path, include
from users.views import AddUserRoles

urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/', include('users.urls')),
    path('user/add-role', AddUserRoles.as_view()),
    path('model/', include('predictor.urls')),
]

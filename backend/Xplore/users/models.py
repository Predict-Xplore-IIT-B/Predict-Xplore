from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin
from django.db import models
from users.managers import UserManager

class User(AbstractBaseUser, PermissionsMixin):
    USER_ROLES = [('admin', 'Admin'),
    ('user', 'User'),]

    username = models.CharField(max_length=255, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=5,choices=USER_ROLES, default='user') # Determines DB Access
    user_roles = models.JSONField(default=list)     # Determines application privilege
    phone_number = models.CharField(max_length=15, unique=True)
    otp = models.CharField(max_length=6, null=True, blank=True)
    otp_expiry = models.DateTimeField(blank=True, null=True)

    # required only for django (DB) admin
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)


    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['email','phone_number']

    objects = UserManager()

    def __str__(self):
        return self.username

    def has_perm(self, perm, obj=None):
        return self.role == 'admin'
    def has_module_perms(self, app_label):
        return self.role == 'admin'

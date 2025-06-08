from django.urls import path
from . import views

urlpatterns = [
    # Registration & verification
    path('register',             views.register.as_view(),              name='register'),
    path('verify-email',         views.verify_email.as_view(),          name='verify-email'),
    path('<str:case>/resend-otp',views.resend_otp.as_view(),           name='resend-otp'),

    # Login, token check & logout
    path('login',                views.LoginView.as_view(),             name='login'),
    path('verify-token',         views.CheckTokenValidity.as_view(),    name='verify-token'),
    path('logout',               views.LogoutView.as_view(),            name='logout'),

    # LDAP integration
    path('verify-ldap',          views.LdapAuth.as_view(),              name='verify-ldap'),
    path('ldap/register',        views.LdapRegisterView.as_view(),      name='ldap-register'),
    path('verify-otp',           views.LoginOTPVerification.as_view(),   name='verify-otp'),

    # Admin-only user CRUD (non-admin users)
    path('admin/users',          views.AdminUserManagementView.as_view(),name='admin-user-management'),

    # Role assignment (if separate)
    path('user/add-role',        views.AddUserRoles.as_view(),           name='add-user-roles'),
]

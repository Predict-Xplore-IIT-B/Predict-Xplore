from django.test import TestCase, Client
from django.urls import reverse
from users.models import User
from django.utils import timezone
import datetime

class UserAuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = reverse('register')  # Ensure your URL patterns have names
        self.verify_email_url = reverse('verify-email')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')

        self.user_data = {
            'username': 'testuser',
            'email': 'testuser@example.com',
            'phone_number': '1234567890',
            'password': 'password123',
            'confirm_password': 'password123',
        }

        # Create a test user
        self.user = User.objects.create(
            username='existinguser',
            email='existinguser@example.com',
            phone_number='0987654321',
            password='password123',
            otp='123456',
            otp_expiry=timezone.now() + datetime.timedelta(minutes=5),
            is_active=True
        )

    def test_register_user(self):
        response = self.client.post(self.register_url, self.user_data)
        self.assertEqual(response.status_code, 201)
        self.assertIn('User registered successfully', response.json()['message'])

    def test_verify_email(self):
        response = self.client.post(self.verify_email_url, {
            'email': self.user.email,
            'otp': self.user.otp,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Email is verified', response.json()['success'])

    def test_login(self):
        response = self.client.post(self.login_url, {
            'email': self.user.email,
            'password': 'password123',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('Login successful', response.json()['message'])

    def test_logout(self):
        # Authenticate the client first
        self.client.force_login(self.user)
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Successfully logged out', response.json()['message'])

class AdminUserManagementTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.admin = User.objects.create(
            username='adminuser',
            email='admin@example.com',
            phone_number='1111111111',
            password='password123',
            role='admin',
            is_active=True,
            is_staff=True
        )
        self.admin.set_password('password123')
        self.admin.save()
        self.user = User.objects.create(
            username='normaluser',
            email='normal@example.com',
            phone_number='2222222222',
            password='password123',
            role='user',
            is_active=True
        )
        self.user.set_password('password123')
        self.user.save()
        self.admin_url = reverse('admin-user-management') if 'admin-user-management' in [u.name for u in self.client.handler._urls.urlpatterns] else '/users/admin/users'

    def test_admin_can_list_non_admin_users(self):
        self.client.force_login(self.admin)
        response = self.client.get(self.admin_url)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(u['username'] == 'normaluser' for u in response.json()))

    def test_admin_can_create_non_admin_user(self):
        self.client.force_login(self.admin)
        data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'phone_number': '3333333333',
            'password': 'password123'
        }
        response = self.client.post(self.admin_url, data)
        self.assertEqual(response.status_code, 201)
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_admin_can_update_non_admin_user(self):
        self.client.force_login(self.admin)
        data = {
            'id': self.user.id,
            'phone_number': '9999999999'
        }
        response = self.client.put(self.admin_url, data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.user.refresh_from_db()
        self.assertEqual(self.user.phone_number, '9999999999')

    def test_admin_can_delete_non_admin_user(self):
        self.client.force_login(self.admin)
        data = {'id': self.user.id}
        response = self.client.delete(self.admin_url, data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(id=self.user.id).exists())

    def test_non_admin_cannot_access_admin_api(self):
        self.client.force_login(self.user)
        response = self.client.get(self.admin_url)
        self.assertEqual(response.status_code, 403)

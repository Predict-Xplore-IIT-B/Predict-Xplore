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

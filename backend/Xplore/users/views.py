import os
import re
import random
import datetime
from django.utils import timezone
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.contrib.auth.hashers import make_password
from django.conf import settings
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from users.models import User
from django.contrib.auth import authenticate, login
from rest_framework.authtoken.models import Token
from users.decorators import role_required
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.exceptions import NotFound
from dotenv import load_dotenv
from rest_framework.parsers import JSONParser
from utils.gmail_api import send_gmail


load_dotenv()


class AdminOnlyView(APIView):
    @role_required(['admin'])
    def get(self, request):
        return Response({'message': 'Welcome, Admin!'})

OTP_DURATION = 5         # minutes


# body text
body_email_verification_otp = """
Hello {}, Your OTP for email verification on Predict Xplore is {}.
This OTP is valid for {} minutes. Do not share it with anyone else.
"""
body_login_otp = """
Hello {}, Your OTP for Predict Xplore Login is {}.
This OTP is valid for {} minutes. Do not share it with anyone else.
"""
#subject headers
subject_verify_email = "Predict Xplore - Email Verification"
subject_login = "Predict Xplore - Login OTP"


# in your views.py

def send_otp(username, email, subject, body):
    otp = random.randint(10000, 99999)
    try:
        body_filled = body.format(username, otp, OTP_DURATION)
        send_gmail(email, subject, body_filled)
    except Exception as e:
        print(f"[WARNING] Gmail API send_otp failed: {e}")
        from rest_framework.exceptions import APIException
        raise APIException("Could not send verification email using Gmail API.")
    return otp


class CheckTokenValidity(APIView):
    def post(self, request):
        email = request.data.get('email')
        token = request.data.get('token')

        if not email or not token:
            return Response("Email and token are required.", status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response("User with the provided email does not exist.", status=status.HTTP_404_NOT_FOUND)

        try:
            user_token = Token.objects.get(user=user)
        except Token.DoesNotExist:
            return Response("No token associated with this user.", status=status.HTTP_401_UNAUTHORIZED)

        if token == user_token.key:
            return Response("Token validated. User is Authenticated.", status=status.HTTP_200_OK)

        return Response("Invalid token.", status=status.HTTP_401_UNAUTHORIZED)
    
    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

class register(APIView):
    def post(self, request):
        # Validation
        username = request.data.get('username')
        email = request.data.get('email')
        phone_number = request.data.get('phone_number')
        password = request.data.get('password')
        confirm_password = request.data.get('confirm_password')
        role = request.data.get('role')
        otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)

        if not all([username, email, phone_number, password, confirm_password]):
            return Response({'error': 'All fields are required.'}, status=400)

        if password != confirm_password:
            return Response({'error': 'Passwords do not match.'}, status=400)

        try:
            validate_email(email)
        except ValidationError:
            return Response({'error': 'Invalid email format.'}, status=400)

        if User.objects.filter(email=email).exists() or User.objects.filter(phone_number=phone_number).exists():
            return Response({'error': 'Email or phone number is already registered.'}, status=409)
        
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username already exists.'}, status=409)

        # Hash the password
        hashed_password = make_password(password)

        otp = send_otp(username, email, subject_verify_email, body_email_verification_otp)
        # Create a new user instance but do not activate it yet (needs OTP verification)
        user = User.objects.create(
            username=username,
            email=email,
            phone_number=phone_number,
            password=hashed_password,
            otp=otp,
            otp_expiry=otp_expiry,
            is_active=False  # Set to False until OTP is verified
        )

        if role in ('admin', 'Admin'):
            user.role = role
            user.is_staff = True

        user.save()

        return Response({'message': 'User registered successfully. Please verify your Email address.'}, status=201)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

class verify_email(APIView):
    def post(self, request):
        otp = request.data.get('otp')
        email = request.data.get('email')

        try:
            # Retrieve the user by email
            user = User.objects.get(email=email)

            # Check if the OTP matches
            if user.otp == otp:
                # check if the OTP has expired
                if user.otp_expiry and user.otp_expiry < timezone.now():
                    return Response({'error': 'OTP has expired.'}, status=400)

                # Activate the user account upon successful OTP verification
                user.is_active = True
                user.otp = None  # Clear the OTP once verified
                user.save()

                return Response({'success':'Email is verified.'}, status=200)
            else:
                return Response({'message':'OTP is incorrect'},status=status.HTTP_400_BAD_REQUEST)

        except User.DoesNotExist:
            return Response({'error': 'User with this email does not exist.'}, status=404)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)

class LoginOTPVerification(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        otp = request.data.get('otp')
        user = request.user  # Use the authenticated user

        if user.otp == otp:
            # check if the OTP has expired
            if user.otp_expiry and user.otp_expiry < timezone.now():
                return Response({'error': 'OTP has expired.'}, status=400)
            user.otp = None  # Clear the OTP once verified
            user.save()

            return Response({'success':'OTP Verified.'},status=200)
        else:
            return Response({'message':'OTP is incorrect'},status=status.HTTP_401_UNAUTHORIZED)

class resend_otp(APIView):
    def post(self, request, case):
        # sends otp again
        email = request.data.get('email')

        # k-v mapping for email/login case
        usecase = {'email':[subject_verify_email, body_email_verification_otp], 'login':[subject_login, body_login_otp]}

        try:
            user = User.objects.get(email=email)

            user.otp = send_otp(user.username, email, usecase[case][0], usecase[case][1])
            user.otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)
            user.save()

        except User.DoesNotExist:
            return Response({'error': 'User with this email address does not exist.'}, status=404)

        return Response({'message':'OTP resent.'},status=200)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


# Login API-> allows the user to login once email verification via OTP is successful

class LoginView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')

        if not username or not password:
            return Response({'error': 'Username and password are required.'}, status=400)

        # Authenticate the user using username and password
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': 'User with this username does not exist.'}, status=404)

        user = authenticate(username=username, password=password)

        if user and user.is_active:
            # Generate a token for the authenticated user
            token, created = Token.objects.get_or_create(user=user)
            user.otp = send_otp(user.username, user.email, subject_login, body_login_otp)
            user.otp_expiry = timezone.now() + datetime.timedelta(minutes=OTP_DURATION)
            user.save()

            user_roles = user.user_roles
            return Response({'token': token.key, 'message': 'Login successful. Login OTP sent.', 'username':user.username,'phone_number':user.phone_number, 'role':user.role, 'user_roles':user_roles}, status=200)

        return Response({'error': 'Invalid credentials or email not verified.'}, status=401)

    def get(self, request):
        return Response({'error': 'Invalid request method.'}, status=status.HTTP_405_METHOD_NOT_ALLOWED)


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure the user is authenticated

    def post(self, request):
        try:
            # Get the token associated with the authenticated user
            user = request.user
            token = Token.objects.get(user=user)

            # delete their reports from file directory once the user logs out
            pattern = re.compile(rf"^{re.escape(user.username+'_')}.*")
    
            directory_path = os.path.join(settings.BASE_DIR,'reports')

            if os.path.exists(directory_path):
                for filename in os.listdir(directory_path):
                    if pattern.match(filename):
                        file_path = os.path.join(settings.BASE_DIR, 'reports', filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {user.username}'s data from file directory.")
                        else:
                            print(f"Skipped (not a file): {file_path}")

            if token:
                token.delete()
                return Response({'message': 'Successfully logged out and token destroyed.'}, status=200)

            raise AuthenticationFailed('Authentication credentials were not provided.')

        except Token.DoesNotExist:
            raise AuthenticationFailed('No token found or invalid token.')

    def get(self, request):
        return Response({'error': 'Invalid request method. Use POST to log out.'}, status=405)

class AddUserRoles(APIView):
    # Admin-only access, to be implemented

    def post(self, request):
        try:
            # JSON Input
            email = request.data.get('email')
            user = User.objects.get(email=email)

            # Parse JSON to get roles to be added
            roles_to_add = request.data.get('roles', [])

            for role in roles_to_add:
                if role not in user.user_roles:
                    user.user_roles.append(role)

            user.save()

            return Response({"message": "Roles added successfully", "roles": user.user_roles}, status=status.HTTP_200_OK)

        except User.DoesNotExist:
            raise NotFound(detail="User not found")
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class AdminUserManagementView(APIView):
    """
    Admins can list, create, update, and delete non-admin users.
    Assumes:
    - User model has a JSONField `user_roles` (default=list).
    - Optional: User model may define `ALLOWED_USER_ROLES = [...]` for validation.
    - role_required decorator checks that request.user.role is in the given list.
    """

    @role_required(['admin'])
    def get(self, request):
        # List all non-admin users
        users = User.objects.filter(role='user')
        user_list = []
        for user in users:
            user_list.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'phone_number': user.phone_number,
                'user_roles': user.user_roles,  # JSONField: likely a list
                'is_active': user.is_active,
            })
        return Response(user_list, status=status.HTTP_200_OK)

    @role_required(['admin'])
    def post(self, request):
        # Create a new non-admin user
        data = request.data
        required_fields = ['username', 'email', 'phone_number', 'password']
        if not all(field in data for field in required_fields):
            return Response({'error': 'Missing required fields.'}, status=status.HTTP_400_BAD_REQUEST)

        username = data['username']
        email = data['email']
        phone_number = data['phone_number']
        password = data['password']

        # Duplicate checks
        if User.objects.filter(email=email).exists() or User.objects.filter(phone_number=phone_number).exists():
            return Response({'error': 'Email or phone number already exists.'}, status=status.HTTP_409_CONFLICT)
        if User.objects.filter(username=username).exists():
            return Response({'error': 'Username already exists.'}, status=status.HTTP_409_CONFLICT)

        # Hash password
        hashed_password = make_password(password)

        # Handle user_roles: expect a list of strings (or empty)
        user_roles = data.get('user_roles', [])
        if user_roles is None:
            user_roles = []
        if not isinstance(user_roles, list):
            return Response({'error': 'user_roles must be a list.'}, status=status.HTTP_400_BAD_REQUEST)

        # Optional validation against allowed roles
        allowed = getattr(User, 'ALLOWED_USER_ROLES', None)
        if allowed is not None:
            invalid = [r for r in user_roles if r not in allowed]
            if invalid:
                return Response(
                    {'error': f'Invalid roles in user_roles: {invalid}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Create user
        user = User.objects.create(
            username=username,
            email=email,
            phone_number=phone_number,
            password=hashed_password,
            role='user',
            is_active=True,
            user_roles=user_roles,
        )
        return Response({'message': 'User created successfully.', 'user_id': user.id}, status=status.HTTP_201_CREATED)

    @role_required(['admin'])
    def put(self, request):
        # Update a non-admin user (by id)
        data = request.data
        user_id = data.get('id')
        if not user_id:
            return Response({'error': 'User id is required.'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=user_id, role='user')
        except User.DoesNotExist:
            return Response({'error': 'User not found or is admin.'}, status=status.HTTP_404_NOT_FOUND)

        # Update simple fields
        for field in ['username', 'email', 'phone_number', 'is_active']:
            if field in data:
                setattr(user, field, data[field])

        # Handle user_roles if present
        if 'user_roles' in data:
            ur = data['user_roles']
            if ur is None:
                ur = []
            if not isinstance(ur, list):
                return Response({'error': 'user_roles must be a list.'}, status=status.HTTP_400_BAD_REQUEST)
            # Optional validation
            allowed = getattr(User, 'ALLOWED_USER_ROLES', None)
            if allowed is not None:
                invalid = [r for r in ur if r not in allowed]
                if invalid:
                    return Response(
                        {'error': f'Invalid roles in user_roles: {invalid}'},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            user.user_roles = ur

        # Handle password change
        if 'password' in data and data['password']:
            user.password = make_password(data['password'])

        user.save()
        return Response({'message': 'User updated successfully.'}, status=status.HTTP_200_OK)

    @role_required(['admin'])
    def delete(self, request):
        # Delete a non-admin user (by id)
        data = request.data
        user_id = data.get('id')
        if not user_id:
            return Response({'error': 'User id is required.'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = User.objects.get(id=user_id, role='user')
            user.delete()
            return Response({'message': 'User deleted successfully.'}, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            return Response({'error': 'User not found or is admin.'}, status=status.HTTP_404_NOT_FOUND)
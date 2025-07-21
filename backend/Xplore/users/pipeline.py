from social_core.exceptions import AuthForbidden
from users.models import User
from django.shortcuts import redirect

def require_verified_user(strategy, details, user=None, *args, **kwargs):
    email = details.get('email')
    name = details.get('fullname') or details.get('first_name', '') + ' ' + details.get('last_name', '')
    if not email:
        raise AuthForbidden('No email provided by Google.')
    try:
        user = User.objects.get(email=email)
        # User exists, redirect to login form with email and name
        return strategy.redirect(f'http://localhost:5173/login?email={email}&name={name}')
    except User.DoesNotExist:
        # User does not exist, redirect to register form with email and name
        return strategy.redirect(f'http://localhost:5173/?email={email}&name={name}')
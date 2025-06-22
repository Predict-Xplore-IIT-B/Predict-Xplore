from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth import get_user_model

class Command(BaseCommand):
    help = 'Create a test user with required fields (username, email, phone_number, password)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            required=True,
            help='Username for the new user',
        )
        parser.add_argument(
            '--email',
            type=str,
            required=True,
            help='Email address for the new user',
        )
        parser.add_argument(
            '--phone_number',
            type=str,
            required=True,
            help='Phone number for the new user',
        )
        parser.add_argument(
            '--password',
            type=str,
            required=True,
            help='Password for the new user',
        )

    def handle(self, *args, **options):
        User = get_user_model()
        username = options['username']
        email = options['email']
        phone = options['phone_number']
        password = options['password']

        if User.objects.filter(username=username).exists():
            self.stdout.write(self.style.WARNING(f"User with username '{username}' already exists."))
            return

        try:
            # Adjust if your create_user signature differs
            user = User.objects.create_user(username=username, email=email, phone_number=phone, password=password)
            user.save()
        except TypeError as e:
            raise CommandError(
                f"Error creating user: {e}\n"
                "Check that create_user accepts: username, email, phone_number, password."
            )
        except Exception as e:
            raise CommandError(f"Unexpected error creating user: {e}")

        self.stdout.write(self.style.SUCCESS(f"User '{username}' created successfully."))

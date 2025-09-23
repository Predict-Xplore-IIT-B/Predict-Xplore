# predictor/tasks.py

import os
import zipfile
import shutil
import subprocess
import logging
import base64
import tempfile  # <-- Import this
from celery import shared_task
from django.contrib.auth import get_user_model
from .models import Container

logger = logging.getLogger(__name__)
User = get_user_model()

@shared_task
def build_container_task(name, description, allowed_users, created_by_id, zip_file_content_b64):
    """
    This Celery task runs the Docker build process in a temporary directory.
    """
    # Use a temporary directory that is automatically created and cleaned up
    with tempfile.TemporaryDirectory() as build_dir:
        try:
            # Decode the zip file content and write it to the temp directory
            zip_file_content = base64.b64decode(zip_file_content_b64)
            zip_path = os.path.join(build_dir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file_content)

            # Unzip the file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(build_dir)
            
            # Build the Docker image
            image_name = f"user_{name.lower().replace(' ', '_')}:latest"
            result = subprocess.run(
                ["docker", "build", "-t", image_name, build_dir],
                check=True, capture_output=True, text=True
            )
            logger.info(f"Docker build success for '{name}': {result.stdout}")
            
            # On successful build, create the Container record in the database
            created_by_user = User.objects.get(id=created_by_id)
            Container.objects.create(
                name=name,
                description=description,
                allowed_users=allowed_users,
                created_by=created_by_user
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build FAILED for '{name}': {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred for '{name}': {e}")

    return f"Task for container '{name}' completed."
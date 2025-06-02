# Use the official Python slim image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /Xplore

# Set OpenCV version
ENV OPENCV_VERSION=4.5.5

# Install system dependencies (including those required for python-ldap)
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential binutils \
    ca-certificates cmake curl \
    ffmpeg gdb gcc g++ git \
    unzip zip wget \
    python3-pip python3-dev python3-numpy python3-distutils \
    python3-setuptools python3-opencv \
    libboost-python-dev libboost-thread-dev libatlas-base-dev \
    libjpeg-dev liblapack-dev libopenblas-dev libopencv-dev \
    libtbb-dev libtesseract-dev libx11-dev libx264-dev \
    locales \
    # Dependencies for python-ldap
    libsasl2-dev libldap2-dev libssl-dev \
    default-jdk \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

# Set Java Home (for ApacheDS)
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV APACHEDS_VERSION=2.0.0.AM27
ENV APACHEDS_HOME=/opt/apacheds-$APACHEDS_VERSION

# Download and Install ApacheDS from the correct URL
RUN wget -O apacheds.deb https://dlcdn.apache.org//directory/apacheds/dist/2.0.0.AM27/apacheds-2.0.0.AM27-amd64.deb \
&& dpkg -i apacheds.deb \
&& rm apacheds.deb

# Ensure ApacheDS binary is executable
RUN chmod +x $(find /opt/ -name "apacheds" | head -n 1)

# Download and extract OpenCV
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv \
    && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \
    && mkdir -p /Xplore/opencv/build

COPY Xplore/requirements.txt /Xplore/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install -r /Xplore/requirements.txt --no-cache-dir

# Copy the rest of the Django project
COPY . /Xplore

# Expose port 8000 for Django and 10389,10636 for LDAP Server
EXPOSE 8000 10389 10636

# Apply Django database migrations
RUN python Xplore/manage.py migrate

# Start ApacheDS properly
RUN /opt/apacheds-2.0.0.AM28-SNAPSHOT/bin/apacheds start default

# Run the Django server, switch to gunicorn in production
CMD ["python", "Xplore/manage.py", "runserver", "0.0.0.0:8000"]
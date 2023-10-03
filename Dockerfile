# Set base image from python version
FROM python:3.10.12

WORKDIR /code

# Copy requirements.txt to /code
COPY requirements.txt .

# Install requirements.txt
RUN pip install -r requirements.txt

# Copy all files to /code
COPY . .

# install libraries required for cv2
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 sudo -y
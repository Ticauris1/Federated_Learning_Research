# 1. Start from an official Python base image
# Using -slim is a good practice to keep the image size smaller
FROM python:3.10-slim

# 2. Set the working directory inside the container
# All future commands will run from /app
WORKDIR /app

# 3. Install system dependencies
# This is needed for OpenCV (libgl1)
# 3. Install system dependencies
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python requirements
# This is done in a separate step to leverage Docker's caching
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project's source code into the container
# This copies everything from the 'src' folder into the container's '/app' folder
COPY src/ .

# 6. Define the command to run your experiment
# This is what runs when the container starts
CMD ["python", "main.py"]
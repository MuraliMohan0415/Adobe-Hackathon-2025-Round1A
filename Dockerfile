
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OCR and ML
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create input/output folders
RUN mkdir -p /app/input /app/output

# Set entrypoint
ENTRYPOINT ["python", "app/src/main.py"]









 
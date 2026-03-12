# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We install these first to leverage Docker cache
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    triton

# Install Mamba specific dependencies
# Note: These require CUDA and can take a while to compile
RUN pip install --no-cache-dir causal-conv1d>=1.2.0 mamba-ssm

# Copy the rest of the application code
COPY . .

# Set environment variables for better performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to run
CMD ["python", "train.py", "--data_dir", "/data"]

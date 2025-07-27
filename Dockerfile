# Use a specific, lightweight Python base image with linux/amd64 platform
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD MODELS ---
# Pre-download both the retriever and re-ranker models to comply with the
# "no internet access" rule during execution.
# We will use Optimum to download a quantized version of the retriever.
RUN python -c "from optimum.onnxruntime import ORTModelForFeatureExtraction; from sentence_transformers import CrossEncoder; \
    ORTModelForFeatureExtraction.from_pretrained('Optimum/bge-large-en-v1.5-onnx-quantized', file_name='model_quantized.onnx', cache_dir='/app/model_cache'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Copy the rest of the application code into the container
COPY . .

# The command that will be executed when the container starts.
CMD ["python3", "main.py"]

# Use Python 3.11 for maximum compatibility with Gradio/OpenEnv deps
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all deployment files into the container
COPY . /app/

# Install Python dependencies from our requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the local 'openenv' folder can be found by Python
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV ENABLE_WEB_INTERFACE=true
ENV PORT=7860

# Expose the Hugging Face Space port
EXPOSE 7860

# Start the application using Uvicorn from the new server/ directory
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

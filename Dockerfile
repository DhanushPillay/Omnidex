FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some numpy/pandas operations)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Render sets PORT env var)
ENV PORT=5000
EXPOSE 5000

# Run the application (using gunicorn for production)
CMD gunicorn app:app --bind 0.0.0.0:$PORT

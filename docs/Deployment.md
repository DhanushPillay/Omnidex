# Deployment Guide

This document explains how to deploy Omnidex to various platforms.

---

## Table of Contents
1. [Local Development](#local-development)
2. [Hugging Face Spaces (Recommended)](#hugging-face-spaces)
3. [Docker](#docker)
4. [Environment Variables](#environment-variables)

---

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/DhanushPillay/Omnidex.git
cd Omnidex

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
set GEMINI_API_KEY=your_api_key_here  # Windows
export GEMINI_API_KEY=your_api_key_here  # Linux/Mac

# Run the server
python app.py
```

### Access
Open `http://localhost:5000` in your browser.

---

## Hugging Face Spaces

Omnidex uses **Hugging Face Spaces** with **Docker** for production deployment.

### Current Deployment
- **URL**: https://huggingface.co/spaces/DecryptVoid/Omnidex
- **Live App**: https://decryptvoid-omnidex.hf.space

### Automatic Deployment

The project uses **GitHub Actions** for CI/CD:

1. Push to `main` branch on GitHub
2. GitHub Actions triggers
3. Code is uploaded to HF Spaces
4. HF Spaces builds Docker image
5. App is live!

### GitHub Actions Workflow

File: `.github/workflows/deploy-hf.yml`

```yaml
name: Deploy to Hugging Face Spaces

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install HF CLI
        run: pip install huggingface_hub
      
      - name: Upload to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python -c "
          from huggingface_hub import HfApi
          api = HfApi()
          api.upload_folder(
              folder_path='.',
              repo_id='DecryptVoid/Omnidex',
              repo_type='space',
              token='$HF_TOKEN',
              ignore_patterns=['*.git*', '.venv/*', '__pycache__/*']
          )
          "
```

### Required Secrets

Add these to your GitHub repository:
- **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | Your Hugging Face access token (with write permission) |

### Required HF Secrets

Add these to your HF Space:
- **Space Settings** → **Repository secrets** → **New secret**

| Secret | Value |
|--------|-------|
| `GEMINI_API_KEY` | Your Google Gemini API key |

---

## Docker

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD gunicorn app:app --bind 0.0.0.0:$PORT
```

### Build and Run Locally

```bash
# Build image
docker build -t omnidex .

# Run container
docker run -p 7860:7860 -e GEMINI_API_KEY=your_key omnidex
```

### Push to Docker Hub

```bash
docker tag omnidex yourusername/omnidex:latest
docker push yourusername/omnidex:latest
```

---

## Environment Variables

| Variable | Required | Description | Where to Set |
|----------|----------|-------------|--------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key | HF Secrets, local env |
| `PORT` | No | Server port (default: 5000/7860) | Dockerfile |
| `SECRET_KEY` | No | Flask session key (auto-generated) | Optional |

### Getting a Gemini API Key

1. Go to https://aistudio.google.com/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key (starts with `AIza...`)

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "No module named 'sklearn'" | Run `pip install scikit-learn` |
| "GEMINI_API_KEY not set" | Set the environment variable |
| Build timeout on HF | sentence-transformers is large, wait longer |
| Port already in use | Change PORT or kill existing process |

### Logs

**Local**: Check terminal output

**HF Spaces**: Go to Space → "Logs" tab

### Memory Issues

If HF Spaces runs out of memory:
1. The semantic model (`sentence-transformers`) uses ~500MB
2. Consider using smaller model or removing it
3. Upgrade to paid HF tier for more RAM

---

## Updating Deployment

### Update HF Spaces

Simply push to GitHub:

```bash
git add .
git commit -m "Update feature"
git push origin main
```

GitHub Actions will automatically deploy to HF Spaces.

### Manual Update

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload DecryptVoid/Omnidex . . --repo-type=space
```

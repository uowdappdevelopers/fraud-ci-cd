# 1. Start from a small official Python image (same minor version as our notebook)
FROM python:3.11-slim

# 2. Sensible Python defaults for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Work inside /app in the image
WORKDIR /app

# 4. Install dependencies FIRST (this layer is cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the code and the model (changes often, so it comes AFTER the install)
COPY app.py .
COPY model/ model/

# 6. Do not run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# 7. Document the port and add a health check
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# 8. The command that starts the service
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-", "app:app"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn for production, fallback to Flask dev server
CMD ["python", "server.py"]

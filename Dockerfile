# Use a multi-stage build to keep the final image size small
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install build dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Final image
FROM python:3.9-slim

# Set a non-root user to improve security
RUN useradd -m appuser
USER appuser

# Set working directory
WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Expose the port the app runs on
EXPOSE 8000

# Set the command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Optionally add a health check (uncomment the lines below)
# HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

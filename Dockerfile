# Start with the Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variable for PORT
ENV PORT=8080

# Use shell form to ensure $PORT is evaluated
CMD exec python3 -m uvicorn main:app --host 0.0.0.0 --port $PORT

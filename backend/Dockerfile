# Use official Python image
FROM python:3.11-slim

# Set workdir inside container
WORKDIR /app
RUN apt-get update && apt-get install -y git && apt-get clean
# Copy requirements first and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy entire backend code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI server with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
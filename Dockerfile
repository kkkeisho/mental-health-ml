
FROM python:3.12-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code and model artifacts into the container
COPY api ./api
COPY models ./models

# 3. Expose the FastAPI port inside the container
EXPOSE 8000

# 4. Command executed when the container starts
#    Launch the FastAPI app defined in api/app.py via uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

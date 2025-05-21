FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
# Cloud Run listens on $PORT (default 8080)
ENV PORT=8080
CMD ["python", "main.py"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir --prefer-binary --timeout 120 --retries 10 -r requirements-app.txt

COPY config ./config
COPY data/app ./data/app
COPY frontend ./frontend
COPY models/tabular ./models/tabular
COPY models/visual_lite ./models/visual_lite
COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

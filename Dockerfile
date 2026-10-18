FROM python:3.15.0-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py train_model.py model.pkl ./
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
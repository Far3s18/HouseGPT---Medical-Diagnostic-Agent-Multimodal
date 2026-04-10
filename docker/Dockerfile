FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*


RUN pip install poetry==2.3.2

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-interaction --no-ansi --no-root

COPY src/ ./src/

RUN mkdir -p logs

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["gunicorn", "house_gpt.api.v1.app:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "logs/access.log", \
     "--error-logfile", "logs/error.log"]
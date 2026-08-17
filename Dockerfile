FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install open-spiel==2.0.2 pokerkit pytest

ENV PYTHONPATH=/app

CMD ["python", "-c", "import open_spiel, pokerkit; print('open_spiel and pokerkit OK')"]

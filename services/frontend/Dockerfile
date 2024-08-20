FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./services/frontend/requirements.txt /tmp/requirements.app.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.app.txt

EXPOSE 80
HEALTHCHECK CMD curl --fail http://localhost:80/_stcore/health
WORKDIR /src

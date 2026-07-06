FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    paho-mqtt \
    requests \
    pydantic \
    numpy \
    scipy \
    grpcio \
    protobuf

COPY services/application/ /app/

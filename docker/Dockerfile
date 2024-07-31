FROM python:3.11-slim

WORKDIR /workspace/langchain_quickstart

RUN pip install --upgrade pip

COPY requirements.txt /workspace/langchain_quickstart/requirements.txt
RUN pip install -r requirements.txt
RUN rm -r requirements.txt
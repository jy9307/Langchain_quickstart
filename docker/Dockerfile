FROM python:3.11-slim

WORKDIR /langchain_quickstart

RUN pip install --upgrade pip

COPY .env /langchain_quickstart/.env
COPY . /langchain_quickstart

COPY requirements.txt /langchain_quickstart/requirements.txt
RUN pip install -r requirements.txt
RUN rm -r requirements.txt

EXPOSE 8000

CMD uvicorn app.server:app --host 0.0.0.0 --port 8000
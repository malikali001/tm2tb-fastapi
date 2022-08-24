FROM python:3.10

RUN apt-get update && apt-get upgrade -y

RUN mkdir /tm2tb-api

COPY tm2tb /tm2tb-api/tm2tb
COPY stopwords /tm2tb-api/stopwords
COPY Pipfile /tm2tb-api
COPY Pipfile.lock /tm2tb-api
COPY api.py /tm2tb-api
COPY LICENSE /tm2tb-api

WORKDIR /tm2tb-api
RUN pip install pipenv \
    && pipenv install --system --deploy

EXPOSE 8000

CMD uvicorn --host 0.0.0.0 --timeout-keep-alive 600 api:app

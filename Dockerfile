
FROM python:3.8

RUN pip install fastapi uvicorn pandas sklearn numpy

COPY abalone_app /abalone_app

ENV PYTHONPATH=/abalone_app

WORKDIR /abalone_app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["abalone_prediction.main:app", "--host", "0.0.0.0"]

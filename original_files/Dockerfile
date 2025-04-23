FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app/scripts

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["./scripts/predict.py", "."]

RUN mkdir -p ../model

COPY ["./model/xgboost_eta=0.3_depth=3_minchild=1_round=100.bin", "../model"]

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "predict:app"]



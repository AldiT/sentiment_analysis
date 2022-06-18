FROM python:3.8

COPY . .

RUN pip install -r requirements.txt

RUN python -m pip install "dask[distributed]" --upgrade

RUN python -m spacy download en_core_web_sm


CMD ["python", "-m", "run"]
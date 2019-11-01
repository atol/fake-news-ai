FROM continuumio/miniconda3

RUN conda install --yes \
    numpy==1.17.2 \
    pandas==0.25.2 \
    scikit-learn==0.21.3

COPY . /usr/src/

CMD [ "python3", "/usr/src/main.py" ]
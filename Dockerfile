FROM continuumio/miniconda3

RUN conda install --yes \
    numpy==1.17.2 

COPY . /usr/src/

CMD [ "python3", "/usr/src/main.py" ]
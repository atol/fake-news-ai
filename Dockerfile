FROM python:3

COPY . /usr/src/

RUN pip install -r /usr/src/requirements.txt

CMD [ "python3", "/usr/src/main.py" ]
FROM python:3

COPY main.py /usr/src/

CMD [ "python3", "/usr/src/main.py" ]

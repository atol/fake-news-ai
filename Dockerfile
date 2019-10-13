FROM python:3

ADD app /usr/src/

CMD [ "python3", "/usr/src/main.py" ]

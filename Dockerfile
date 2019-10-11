FROM python:3

COPY main.py classifiers.py /usr/src/

CMD [ "python3", "/usr/src/main.py" ]

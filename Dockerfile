FROM python:3.9

COPY . /mask_recognition
WORKDIR /mask_recognition

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python main.py

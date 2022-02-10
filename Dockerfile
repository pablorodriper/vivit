FROM tensorflow/tensorflow:latest-gpu

ADD . /app

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

#RUN pip install git+https://github.com/huggingface/huggingface_hub.git@main


FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-gpu-py36-cu100-ubuntu16.04

LABEL maintainer=""

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

COPY src /app

ENTRYPOINT [ "python3" ]

CMD [ "/app/main.py" ]
FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-training:1.2.0-gpu-py36-cu100-ubuntu16.04

LABEL maintainer=""

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt

COPY src /app

WORKDIR /app

ENTRYPOINT [ "python3" ]

CMD [ "/app/main.py" ]
FROM ubuntu:20.04
MAINTAINER Daria Bakalets
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "app.py"]


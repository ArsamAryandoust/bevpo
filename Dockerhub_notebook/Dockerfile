FROM ubuntu:latest

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y build-essential python3 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
RUN pip3 install jupyter
RUN pip3 install bevpo==0.1.2.1


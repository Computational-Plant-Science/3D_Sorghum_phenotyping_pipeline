FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/code
WORKDIR /opt/code


RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    aptitude \
    libboost-all-dev \
    python3-setuptools \
    python3-pip \
    python3 \
    cmake-gui \
    software-properties-common \
    nano

     

RUN pip3 install --upgrade pip && \
    pip3 install numpy \
    matplotlib \
    pytest \
    open3d \
    openpyxl \
    click \
    PyYAML


RUN chmod +x /opt/code/shim.sh 

ENV PYTHONPATH=$PYTHONPATH:/opt/code/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/code/



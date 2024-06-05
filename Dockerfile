FROM ubuntu:20.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/code
WORKDIR /opt/code


RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    aptitude \
    libglu1-mesa-dev \
    mesa-common-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libboost-all-dev \
    python3-setuptools \
    python3-pip \
    python3 \
    python3-tk \
    python3-pil.imagetk \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    cmake-gui \
    freeglut3-dev \
    freeglut3 \
    libopengl0 -y \
    mesa-utils \
    software-properties-common \
    libcairo2 \
    python3-cairo \
    nano \
    xorg-dev


RUN cd /opt/code/compiled/ && rm -rf Release && mkdir Release && cd Release && cmake -DCMAKE_BUILD_TYPE=Release ..   && make


RUN pip3 install --upgrade pip && \
    pip3 install numpy \
    numexpr \
    matplotlib \
    mahotas \
    plyfile \
    psutil \
    pandas \
    pytest \
    coverage \
    coveralls \
    open3d \
    openpyxl \
    click \
    PyYAML \
    imutils




RUN chmod +x /opt/code/shim.sh 

ENV PYTHONPATH=$PYTHONPATH:/opt/code/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/code/



# To build, navigate to the *root* of this repo and run:
#   docker build . -t tonytwang/pii:main
# You may also want to push the image to Docker Hub:
#   docker push tonytwang/pii:main

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y \
  curl \
  git \
  sudo \
  tmux \
  unzip \
  uuid-runtime \
  vim \
  wget \
  # python
  python3.10-venv \
  # ACDC
  build-essential \
  graphviz \
  graphviz-dev \
  libgl1-mesa-glx \
  python3.10-dev \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Create virtualenv and 'activate' it by adjusting PATH.
# See https://pythonspeed.com/articles/activate-virtualenv-dockerfile/.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pip requirements
RUN pip install --no-cache-dir --upgrade pip setuptools
COPY requirements.txt /downloads/requirements.txt
RUN pip install --no-cache-dir -r /downloads/requirements.txt

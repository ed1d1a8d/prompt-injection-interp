# To build, navigate to the *root* of this repo and run:
#   docker build . -f docker/Dockerfile -t tonytwang/pii:main
# You may also want to push the image to Docker Hub:
#   docker push tonytwang/pii:main

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y \
  curl \
  git \
  htop \
  nginx \
  openssh-server \
  python3.10 \
  python3-pip \
  sudo \
  tmux \
  unzip \
  uuid-runtime \
  vim \
  wget \
  # ACDC
  build-essential \
  graphviz \
  graphviz-dev \
  libgl1-mesa-glx \
  python3.10-dev \
  # Clean up
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Add mier user with sudo privileges
RUN adduser --disabled-password --gecos "" mier
RUN usermod -aG sudo mier
RUN echo "mier ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/mier

# Add .local/bin to mier PATH. Needed for pip installs.
RUN echo "export PATH=\$PATH:/home/mier/.local/bin" >> /home/mier/.bashrc

# Switch to mier user
USER mier
WORKDIR /home/mier

# Install jupyerlab globally for runpod to work
RUN pip install --no-cache-dir --upgrade --no-warn-script-location pip setuptools
RUN pip install --no-cache-dir --no-warn-script-location jupyterlab

# Install micromamba
ARG PREFIXLOCATION=/home/mier/.micromamba
ARG MAMBA_ROOT_PREFIX=/home/mier/.micromamba
ARG CONDA_FORGE_YES=yes
RUN curl -L -o install_micromamba.sh micro.mamba.pm/install.sh && \
  bash install_micromamba.sh && \
  rm install_micromamba.sh
# RUN . /home/mier/.bashrc

# Create micromamba environment
RUN /home/mier/.local/bin/micromamba env create -y -n pii python=3.10

# Install pip requirements
RUN /home/mier/.local/bin/micromamba run -n pii \
  pip install --no-cache-dir --upgrade pip setuptools
COPY --chown=mier requirements.txt downloads/requirements.txt
RUN /home/mier/.local/bin/micromamba run -n pii \
  pip install --no-cache-dir -r downloads/requirements.txt

# Install vscode server
FROM base AS vscode
RUN curl -Lk \
  'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64'\
  --output downloads/vscode_cli.tar.gz
RUN tar -xf downloads/vscode_cli.tar.gz

FROM vscode AS final
COPY --from=vscode --chown=mier /home/mier/code code
COPY --chown=mier docker/start.sh downloads/start.sh
CMD ["downloads/start.sh"]

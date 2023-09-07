# To build, navigate to the *root* of this repo and run:
#   DOCKER_BUILDKIT=1 docker build . -f docker/Dockerfile -t tonytwang/pii:main --secret id=hf_token,src=docker/.hf_token
# You may also want to push the image to Docker Hub:
#   docker push tonytwang/pii:main

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# Install utilities
RUN apt-get update -q \
  && apt-get install -y \
  curl \
  git \
  git-lfs \
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
# https://mamba.readthedocs.io/en/latest/micromamba-installation.html#manual-installation
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN mkdir -p .local/bin && mv bin/micromamba .local/bin && rmdir bin
RUN .local/bin/micromamba shell init -s bash -p /home/mier/.micromamba
RUN .local/bin/micromamba config append channels conda-forge \
  && .local/bin/micromamba config append channels nodefaults \
  && .local/bin/micromamba config set channel_priority strict

# Create micromamba environment
ARG MAMBA_ROOT_PREFIX=/home/mier/.micromamba
RUN /home/mier/.local/bin/micromamba env create -y -n pii python=3.10

# Install pip requirements
RUN /home/mier/.local/bin/micromamba run -n pii \
  pip install --no-cache-dir --upgrade pip setuptools
COPY --chown=mier requirements.txt downloads/requirements.txt
RUN /home/mier/.local/bin/micromamba run -n pii \
  pip install --no-cache-dir -r downloads/requirements.txt

# Set up git credential store
RUN git config --global credential.helper store


FROM base AS hf-downloads

COPY --chown=mier docker/dl_hf_models.py downloads/dl_hf_models.py
RUN --mount=type=secret,id=hf_token,uid=1000 \
  /home/mier/.local/bin/micromamba run -n pii \
  python downloads/dl_hf_models.py


FROM base AS final

# Copy huggingface cache
COPY --from=hf-downloads /home/mier/.cache/huggingface /home/mier/.cache/huggingface

# Copy start script
COPY --chown=mier docker/start.sh downloads/start.sh
CMD ["downloads/start.sh"]

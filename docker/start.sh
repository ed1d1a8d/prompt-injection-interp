#!/bin/bash
# Adapted from https://github.com/runpod/containers/blob/main/container-template/start.sh

set -e  # Exit the script if any statement returns a non-true return value

if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
    service ssh start
fi

sleep infinity

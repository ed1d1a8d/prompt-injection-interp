#!/bin/bash

black --check pii
black --check notebooks
isort --check pii

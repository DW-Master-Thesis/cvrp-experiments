#!/bin/bash

python3 -m venv venv
source venv/bin/activate

pip install -r requirements-dev.txt
pip install -e .

pre-commit install --install-hooks

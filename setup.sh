#!/bin/bash

git submodule update --init --recursive

if [ -d "venv" ]; then
echo "Virtual environment already exists."
else
python -m venv venv
fi

. venv/bin/activate
pip install -r requirements+cuda.txt
pip install submodules/diff-gaussian-rasterization
deactivate




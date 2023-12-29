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

if [ ! -d "data" ]; then
mkdir -p "data"
fi

if [ ! -d "data/garden" ]; then
mkdir -p "data/garden"
wget https://huggingface.co/datasets/Subuday/GaussianSplatting/resolve/main/garden.tgz
tar -xvf garden.tgz
rm garden.tgz
rm ._garden
mv garden data/
fi

git config --global user.name "Maksym Sutkovenko"
git config --global user.email "mr.gigant977@gmail.com"
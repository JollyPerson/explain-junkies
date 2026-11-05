# Train-test loop for models with OOD Classes

## Getting started

To get started, intsall dependencies from requirements.txt:

`pip3 install -r requirements.txt`

Then run the main file with:

`python3 main.py <pretrain|train|measure|all>`

## TODO:

This repository has been developed for CIFAR/IMAGENET. This version is configured to exit early if those arent present. Your task is to refactor this repository so that it can run locally with MNIST, and perhaps to generate some OOD data for MNIST with which to trian.
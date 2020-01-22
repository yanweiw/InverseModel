#!/bin/bash

python train.py --tag world --seed 116 --epoch 20
python train.py --tag world --seed 117 --epoch 20
python train.py --tag world --seed 118 --epoch 20


python train.py --tag joint --seed 119 --epoch 20
python train.py --tag joint --seed 120 --epoch 20
python train.py --tag joint --seed 121 --epoch 20


python train.py --tag pixel --seed 122 --epoch 20
python train.py --tag pixel --seed 123 --epoch 20
python train.py --tag pixel --seed 124 --epoch 20

#!/bin/bash

python train.py --tag world --seed 125 --epoch 20
python train.py --tag world --seed 126 --epoch 20
# python train.py --tag world --seed 118 --epoch 20


python train.py --tag joint --seed 127 --epoch 20
python train.py --tag joint --seed 128 --epoch 20
# python train.py --tag joint --seed 121 --epoch 20


python train.py --tag pixel --seed 129 --epoch 20
python train.py --tag pixel --seed 130 --epoch 20
# python train.py --tag pixel --seed 124 --epoch 20

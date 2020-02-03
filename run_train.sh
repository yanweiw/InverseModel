#!/bin/bash

# python train.py --tag world --seed 125 --epoch 20
# python train.py --tag world --seed 126 --epoch 20
# python train.py --tag world --seed 118 --epoch 20


# python train.py --tag joint --seed 127 --epoch 20
# python train.py --tag joint --seed 128 --epoch 20
# python train.py --tag joint --seed 121 --epoch 20


# python train.py --tag pixel --seed 129 --epoch 20
# python train.py --tag pixel --seed 130 --epoch 20
# python train.py --tag pixel --seed 124 --epoch 20


for SEED in 0 1 2 3
do
  python train.py --tag wpoke --seed $SEED
done

for SEED in 5 6 7 8 9
do
  python train.py --tag world --seed $SEED
done

for SEED in 10 11 12 13 14
do
  python train.py --tag pixel --seed $SEED
done

for SEED in 15 16 17 18 19
do
  python train.py --tag joint --seed $SEED
done 

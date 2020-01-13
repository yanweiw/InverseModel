# Data description

image_0-7: arm no reset

image_10-17: arm reset but occasionally unreachable

image_20-27, 30-37: arm reset always reachable

image_28-29: long range data; 28 all, 29 first 1k data no collision

image_38-39: even longer range data; 38 all, 39 first 1k data no collision

image_40_48: larger box

Learning (normalize data to (-1, 1), mean: [-0.908,  0.356, -0.908], std: [0.144, 0.069, 0.158]):

image_20: 0.07556088717675045
image_21: 0.03013499572977911 (training)
image_28: 0.0963768486964377
image_38: 0.13977662882761696

Base:

image_20: 0.13977662882761696
image_21: 0.09778188921837593
image_28: 0.12106623075863897
image_38: 0.12375992358185979

Learning with 100K

image_20: 0.05323721414637045; 0.02910992555545167
image_21: 0.01757739567784716 (training); 0.01640840114215274

Learning (normalize data to (-5, 5), mean: [ 0.001, -0.001, -0.001], std: [0.998, 1.022, 1.012]):

image_20: 0.03995610861120316
image_21: 0.02433286132548355 (training)

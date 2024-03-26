#!/bin/bash

num_test=64

n_corrector=0
delta=0.

dataset=flowers
# dataset=celeba
task=inpainting


method=replacement
# batch_size=32

# for start_fraction in 0.5 0.7 1.
# do
#     for noise in true false
#     do
#         python main.py --config=config.py:$dataset,$task,$method --mode=eval --config.testing.num_test=$num_test --config.conditioning.noise=$noise --config.conditioning.start_fraction=$start_fraction --config.conditioning.n_corrector=$n_corrector --config.conditioning.delta=$delta --config.testing.batch_size=$batch_size --config.testing.seed=1
#     done
# done


# start_fraction=1.
# noise=true
# for seed in 0 1
# do
#     for n_corrector in 1 2 3 5
#     do
#         for delta in 1. 0.5 0.1 0.05 0.01
#         do
#             python main.py --config=config.py:$dataset,$task,$method --mode=eval --config.testing.num_test=$num_test --config.conditioning.noise=$noise --config.conditioning.start_fraction=$start_fraction --config.conditioning.n_corrector=$n_corrector --config.conditioning.delta=$delta  --config.testing.batch_size=$batch_size --config.testing.seed=$seed
#         done
#     done
# done


method=reconstruction_guidance
batch_size=8

for start_fraction in 0.5 0.7 1.
do
    for gamma in 0.1 0.5 1. 2. 5. 10.
    do
        python main.py --config=config.py:$dataset,$task,$method --mode=eval --config.testing.num_test=$num_test --config.conditioning.start_fraction=$start_fraction  --config.conditioning.gamma=$gamma --config.conditioning.n_corrector=$n_corrector --config.conditioning.delta=$delta --config.testing.batch_size=$batch_size --config.testing.seed=0
    done
done


# start_fraction=0.5
# for seed in 0 1
# do
#     for n_corrector in 1 2 3 5
#     do
#         for delta in 1. 0.5 0.1 0.05 0.01
#         do
#             python main.py --config=config.py:$dataset,$task,$method --mode=eval --config.testing.num_test=$num_test --config.conditioning.start_fraction=$start_fraction --config.conditioning.n_corrector=$n_corrector --config.conditioning.delta=$delta --config.testing.batch_size=$batch_size --config.testing.seed=$seed
#         done
#     done
# done
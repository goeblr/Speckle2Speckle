#!/bin/bash

cd src

python3 train.py \
    --train-dir <folder_training_images> \
    --train-size 1000 \ 
    --valid-dir <folder_validation_images> \ 
    --valid-size 100 \ 
    --ckpt-save-path ckpts \
    --nb-epochs 1000 \
    --batch-size 5 \ 
    --report-interval 25 \
    --loss l2+interfacePSF \
    --noise-type intrinsic \
    --crop-size 128 \
    --plot-stats \
    --cuda \
    --average-validation-targets \
    --ckpt-overwrite \
    --seed 1000 \
    --learning-rate 3e-05 \
    --psf_gauss_sigma_x 4 \
    --psf_gauss_sigma_y 0.8 \
    --interface_weight 500 \ 
    --interface_gauss_sigma 5
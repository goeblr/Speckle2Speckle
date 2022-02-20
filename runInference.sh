#!/bin/bash

mkdir -p outputImages/invivo
mkdir -p outputImages/experimental
mkdir -p outputImages/simulated

cd src
python3 apply.py --input-data ../data/invivo --output-data ../outputImages/invivo --load-ckpt ../data/network.pt --cuda --blending-factor 0
python3 apply.py --input-data ../data/phantom --output-data ../outputImages/experimental --load-ckpt ../data/network.pt --cuda --blending-factor 0
python3 apply.py --input-data ../data/test-set_simulated --output-data ../outputImages/simulated --load-ckpt ../data/network.pt --cuda --blending-factor 0

# This runs inference on crops of the images, and also saves a comparison image for each input
# python3 test.py --noise-type intrinsic \
#       --crop-size 480 \
#       --show-output 151 \
#       --seed 10 \
#       --load-ckpt \
#       ../data/network.pt \
#       --cuda \
#       --data ../data/invivo --output-path ../outputImages/invivo
# python3 test.py --noise-type intrinsic \
#       --crop-size 480 \
#       --show-output 151 \
#       --seed 10 \
#       --load-ckpt \
#       ../data/network.pt \
#       --cuda \
#       --data ../data/phantom --output-path ../outputImages/experimental
# python3 test.py --noise-type intrinsic \
#       --crop-size 480 \
#       --show-output 151 \
#       --seed 10 \
#       --load-ckpt \
#       ../data/network.pt \
#       --cuda \
#       --average-validation-targets \
#       --data ../data/test-set_simulated --output-path ../outputImages/simulated
# Speckle2Speckle

This implementation of Speckle2Speckle is based on code of Joey Litalien
https://github.com/joeylitalien/noise2noise-pytorch

## apply network to test-set, phantom and invivo
runInference.sh

## apply other methods
evaluation/otherMethods/otherMethods_experimental.m
evaluation/otherMethods/otherMethods_invivo.m
evaluation/otherMethods/otherMethods_simulated.m

## compute metrics & create plots
cd src
python3 compute_metrics.py

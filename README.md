# Speckle2Speckle

This implementation of Speckle2Speckle is based on code of Joey Litalien
https://github.com/joeylitalien/noise2noise-pytorch

## License
The files in the folder `src` can be used under the MIT license (see the `LICENSE` file there).

The implementations of the comparison methods (`evaluation/otherMethods/SRAD/` and `evaluation/otherMethods/OBNLMpackage/`) follow their own licensing.

All other content including the images can only be used for research purposes.

## Unpack Zips
Unpack the following ZIPs in their folders:

* `data/test-set_simulated.zip`: It contains the test-set of the simulated data.
* `outputImages/otherMedhods_simulated.zip`: Contains the precomputed output of the other methods on the simulated test-set

## Apply Network to test-set, phantom and invivo
    runInference.sh

## Apply Other Methods
Precomputed images are included as the application takes significant time.

    evaluation/otherMethods/otherMethods_experimental.m
    evaluation/otherMethods/otherMethods_invivo.m
    evaluation/otherMethods/otherMethods_simulated.m

## Compute Metrics & Create Plots
    cd src
    python3 compute_metrics.py

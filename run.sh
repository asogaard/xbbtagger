#!/bin/bash
# Run reweighting.py, preparing.py, and neural network classifier training.

if [[ ! "$(pwd -P)" == *xbbtagger ]]; then
    echo "[ERROR] Please run from the xbbtagger directory."
    return 1
elif [ ! -d Preprocessing ] || [ ! -d Training ]; then 
    echo "[ERROR] Unable to find the Preprocessing and/or Training directories."
    return 1
elif ! type conda > /dev/null 2>&1; then
    echo "[ERROR] Conda is not installed."
    return 1
elif [ -z "$(conda env list | grep xbbtagger)" ]; then
    echo "[ERROR] No xbbtagger environment was installed in conda."
fi

# Activate conda environment
active_env="$(conda env list | grep "*" | sed 's/ .*//g')"
target_env="$(conda env list | grep xbbtagger | tail -1 | sed 's/ .*//g')"
if  [[ "$active_env" != *"xbbtagger"* ]] && [ "$active_env" != "$target_env" ]; then
    if [[ ! $active_env =~ root|base ]]; then
	echo "Deactivating environment $active_env"
	source deactivate
    fi
    echo "Activating environment $target_env"
    source activate $target_env
fi

# Save reference to current directory
mypwd="$(pwd -P)"

# Perform preprocessing
cd Preprocessing
mkdir -p output
echo "-- Removing old output files"
rm -f output/Weight_{0,1}.h5
rm -f output/prepared_sample_{no_scaling_,}v2.h5

echo "-- Running reweighting.py"
python reweighting.py  --pt-flat | tee log_reweighting_0.out
python reweighting.py            | tee log_reweighting_1.out

echo "-- Running preparing.py"
python preparing.py              | tee log_preparing.out

if [ ! -f output/Weight_0.h5 ] || [ ! -f output/Weight_1.h5 ] || [ ! -f output/prepared_sample_v2.h5 ]; then
    echo "[ERROR] One of expected output files from preprocessing were not found."
    cd -P $mypwd
    return 1
fi

# Perform training
cd ../Training
echo "-- Running btagging_nn.py"
KERAS_BACKEND=tensorflow python btagging_nn.py --input_file ../Preprocessing/output/prepared_sample_v2.h5 --batch_size=8192 --patience=5

# @TODO: Check Keras output

# Return
echo "-- Done"
cd -P $mypwd

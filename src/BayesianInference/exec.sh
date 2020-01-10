#!/bin/bash

###########
#  guide  #
###########

## DATASET_NAME    Supported: mnist
## N_SAMPLES       Number of input samples
## LR              Learning rate
## EPOCHS          Number of epochs
## DEVICE          Run on device "cpu" or "cuda"
## SEED            Random initialization seed

#######################################
# settings -> comment unwanted lines! #
#######################################

# === vi_bnn === #
SCRIPT="vi_bnn"
DATASET_NAME="mnist"
N_SAMPLES="10000"
LR="0.0001"
EPOCHS="30"
DEVICE="cpu"

# === hidden_vi_bnn === #
SCRIPT="hidden_vi_bnn"
DATASET_NAME="mnist"
N_INPUTS="10000"
N_SAMPLES="1000"
EPOCHS="30"
LR="0.01"
DEVICE="cpu"

# === hmc_bnn === #
#SCRIPT="hmc_bnn"
#DATASET_NAME="mnist"

###########
# execute #
###########

### launch from anywhere on cluster
if [ $HOSTNAME != "zenbook" ] ; then
  cd ~/adversarial_examples/src/
fi

## activate environment
if [ "$DEVICE" == "cpu" ]; then
  source ~/virtualenvs/venv/bin/activate
elif [ "$DEVICE" == "gpu" ]; then
  conda activate tensorflow-gpu
fi

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_${DATASET_NAME}_${SCRIPT}_out.txt"

## run script
if [ $SCRIPT = "vi_bnn" ]; then
  python3 "BayesianInference/vi_bnn.py" $DATASET_NAME $N_SAMPLES $LR $EPOCHS $DEVICE > $OUT
elif [ $SCRIPT = "hidden_vi_bnn" ]; then
  python3 "BayesianInference/hidden_vi_bnn.py" $DATASET_NAME $N_INPUTS $N_SAMPLES $EPOCHS $LR $DEVICE > $OUT
fi

## deactivate environment
if [ "$DEVICE" == "cpu" ]; then
  deactivate
elif [ "$DEVICE" == "gpu" ]; then
  conda deactivate
fi
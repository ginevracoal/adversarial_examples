#!/bin/bash

#######################################
# settings -> comment unwanted lines! #
#######################################

# === vi_bnn === #
SCRIPT="vi_bnn"
DATASET_NAME="mnist"
LR="0.002"
EPOCHS="30"
DEVICE="cpu"
N_SAMPLES="10000"
SEED=0

# === hmc_bnn === #
#SCRIPT="hmc_bnn"
#DATASET_NAME="mnist"
#LR="0.002"
#EPOCHS="30"
#DEVICE="cpu"
#N_SAMPLES="10000"
#SEED=0

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
  python3 "BayesianInference/vi_bnn.py" $DATASET_NAME $N_SAMPLES $LR $EPOCHS $DEVICE $SEED > $OUT
#elif [ $SCRIPT = "hm_bnn" ]; then
fi

## deactivate environment
if [ "$DEVICE" == "cpu" ]; then
  deactivate
elif [ "$DEVICE" == "gpu" ]; then
  conda deactivate
fi
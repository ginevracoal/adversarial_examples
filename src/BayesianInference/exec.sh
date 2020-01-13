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
#SCRIPT="vi_bnn"
#DATASET_NAME="mnist"
#N_SAMPLES="10000"
#LR="0.0001"
#EPOCHS="30"
#DEVICE="cpu"

# === hidden_vi_bnn === #
SCRIPT="hidden_vi_bnn"
DATASET_NAME="mnist"
N_INPUTS="60000"
N_SAMPLES="20"
EPOCHS="400"
LR="0.02"
DEVICE="cuda"

# === hmc_bnn === #
#SCRIPT="hmc_bnn"
#DATASET_NAME="mnist"
#N_INPUTS="1000"
#WARMUP="100"
#N_CHAINS="4"
#N_SAMPLES="100"
#DEVICE="cpu"


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
elif [ "$DEVICE" == "cuda" ]; then
  source ~/virtualenvs/venv_gpu/bin/activate
fi

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_${DATASET_NAME}_${SCRIPT}_out.txt"

## run script
if [ $SCRIPT = "vi_bnn" ]; then
  python3 "BayesianInference/vi_bnn.py" $DATASET_NAME $LR $EPOCHS $DEVICE > $OUT
elif [ $SCRIPT = "hidden_vi_bnn" ]; then
  python3 "BayesianInference/hidden_vi_bnn.py" --n_samples=$N_SAMPLES --dataset_name=$DATASET_NAME --n_inputs=$N_INPUTS --n_epochs=$EPOCHS --lr=$LR --device=$DEVICE > $OUT
elif [ $SCRIPT = "hmc_bnn" ]; then
  python3 "BayesianInference/hmc_bnn.py" --dataset_name=$DATASET_NAME --n_inputs=$N_INPUTS --n_chains=$N_CHAINS --warmup=$WARMUP --device=$DEVICE > $OUT
fi

## deactivate environment
deactivate
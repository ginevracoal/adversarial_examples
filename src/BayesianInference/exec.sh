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
N_INPUTS="10000"
N_SAMPLES="10"
EPOCHS="200"
LR="0.0002"
DEVICE="cuda"

# === hmc_bnn === #
#SCRIPT="hmc_bnn"
#DATASET_NAME="mnist"
#N_INPUTS="10000"
#WARMUP="10"
#N_CHAINS="4"
#N_SAMPLES="1000"
#DEVICE="cpu"
#N_STEPS="100"


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
  python3 "BayesianInference/vi_bnn.py" $DATASET_NAME $LR $EPOCHS $DEVICE &> $OUT
elif [ $SCRIPT = "hidden_vi_bnn" ]; then
  python3 "BayesianInference/hidden_vi_bnn.py" --samples=$N_SAMPLES --dataset=$DATASET_NAME --inputs=$N_INPUTS --epochs=$EPOCHS --lr=$LR --device=$DEVICE &> $OUT
elif [ $SCRIPT = "hmc_bnn" ]; then
  python3 "BayesianInference/hmc_bnn.py" --dataset=$DATASET_NAME --inputs=$N_INPUTS --chains=$N_CHAINS --warmup=$WARMUP --device=$DEVICE --steps=$N_STEPS &> $OUT
fi

## deactivate environment
deactivate
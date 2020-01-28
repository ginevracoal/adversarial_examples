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

# === hidden_bnn === #
#SCRIPT="hidden_bnn"
#DATASET_NAME="mnist"
#N_INPUTS="30000"
#EPOCHS="200"
#LR="0.002"
#DEVICE="cpu"

# === hidden_vi_bnn === #
#SCRIPT="hidden_vi_bnn"
#DATASET_NAME="mnist"
#N_INPUTS="100"
#N_SAMPLES="5"
#EPOCHS="100"
#LR="0.00002"
#DEVICE="cpu"

# === hmc_bnn === #
#SCRIPT="hmc_bnn"
#DATASET_NAME="mnist"
#N_INPUTS="10000"
#WARMUP="1000"
#N_CHAINS="8"
#N_SAMPLES="1"
#DEVICE="cpu"

# === scatterplot == #
#SCRIPT="scatterplot"
#DEVICE="cpu"
#DATASET_NAME="mnist"

# === gridplot == #
SCRIPT="gridplot"
DEVICE="cpu"
DATASET_NAME="mnist"
INPUTS="100"

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
if [ $SCRIPT = "hidden_bnn" ]; then
  python3 "BayesianInference/hidden_bnn.py" --dataset=$DATASET_NAME --inputs=$N_INPUTS --epochs=$EPOCHS --lr=$LR --device=$DEVICE &> $OUT
elif [ $SCRIPT = "hidden_vi_bnn" ]; then
  python3 "BayesianInference/hidden_vi_bnn.py" --samples=$N_SAMPLES --dataset=$DATASET_NAME --inputs=$N_INPUTS --epochs=$EPOCHS --lr=$LR --device=$DEVICE &> $OUT
elif [ $SCRIPT = "hmc_bnn" ]; then
  python3 "BayesianInference/hmc_bnn.py" --dataset=$DATASET_NAME --inputs=$N_INPUTS --chains=$N_CHAINS --warmup=$WARMUP --device=$DEVICE &> $OUT
elif [ $SCRIPT = "scatterplot" ]; then
  python3 "BayesianInference/plots/scatterplot_accuracy_robustness_tradeoff.py" --dataset=$DATASET_NAME --device=$DEVICE &> $OUT
elif [ $SCRIPT = "gridplot" ]; then
  python3 "BayesianInference/plots/gridplot_exp_loss_gradients_norms.py" --dataset=$DATASET_NAME --device=$DEVICE --inputs=$INPUTS &> $OUT
fi

## deactivate environment
deactivate
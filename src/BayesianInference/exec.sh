#!/bin/bash

##############
#  settings  #
##############

DATASET_NAME="mnist"
N_INPUTS="30000"
EPOCHS="200"
LR="0.002"
DEVICE="cuda"
N_SAMPLES="5"
WARMUP="1000"
N_CHAINS="8"

#############
# execution #
#############

### launch from anywhere on server
cd ~/adversarial_examples/src/

## activate gpu environment
source ~/virtualenvs/venv_gpu/bin/activate

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
OUT="${RESULTS}${TIME}_${DATASET_NAME}_out.txt"

## run script

#python3 "BayesianInference/hidden_bnn.py" --dataset=$DATASET_NAME --inputs=$N_INPUTS --epochs=$EPOCHS --lr=$LR --device=$DEVICE &> $OUT
#python3 "BayesianInference/hidden_vi_bnn.py" --samples=$N_SAMPLES --dataset=$DATASET_NAME --inputs=$N_INPUTS --epochs=$EPOCHS --lr=$LR --device=$DEVICE &> $OUT
#python3 "BayesianInference/hmc_bnn.py" --dataset=$DATASET_NAME --inputs=$N_INPUTS --chains=$N_CHAINS --warmup=$WARMUP --device=$DEVICE &> $OUT
#python "BayesianInference/adversarial_attacks.py" --dataset=$DATASET_NAME --device=$DEVICE &> $OUT
python "BayesianInference/loss_gradients.py" --dataset=$DATASET_NAME --device=$DEVICE --inputs=$N_INPUTS &> $OUT
#python3 "BayesianInference/plots/scatterplot_accuracy_robustness_tradeoff.py" --dataset=$DATASET_NAME --device=$DEVICE &> $OUT

# deactivate environment
deactivate
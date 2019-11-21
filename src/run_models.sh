#!/bin/bash

###########
#  guide  #
###########
##  DATASET_NAME    Supported: mnist,cifar.
##  TEST            If True only takes a few samples.
##  ATTACK          Supported: None, fgsm, pgd, deepfool, carlini, newtonfool, virtual, spatial, zoo, boundary
##  N_PROJ_LIST     Supported: lists containing 0,..,15. Default for training is [15], default for testing is [6,9,12,15].
##  SIZE_PROJ_LIST  Supported: list containing 8, 12, 16, 20. Default is [8 12 16 20].
##  PROJ_MODE       Supported for randens, parallel_randens: flat, channels, one_channel, grayscale.
##                  Supported for randreg: no_projections, loss_on_projections, projected_loss, loss_on_perturbations.
## ENSEMBLE_SIZE    Number of ensemble models

#######################################
# settings -> comment unwanted lines! #
#######################################

# === baseline === #
SCRIPT="baseline"
DATASET_NAME="mnist"
TEST="False"
DEVICE="cpu"
SEED=0

# === randens === #
#SCRIPT="randens"
#DATASET_NAME="mnist"
#TEST="False"
#N_PROJ_LIST=[5] #[6,9,12,15]
#SIZE_PROJ_LIST=[8] #[8,12,16,20]
#PROJ_MODE="channels"
#ATTACK="fgsm"
#EPS=0.3
#DEVICE="cpu"

# === parallel_random_ensemble === #
#SCRIPT="parallel_randens"
#DATASET_NAME="mnist"
#TEST="False"
#N_PROJ=5
#SIZE_PROJ_LIST=[20] #[8,12,16,20]
#PROJ_MODE="channels"
#DEVICE="cpu"

# === randreg === #
#SCRIPT="randreg"
#DATASET_NAME="mnist"
#TEST="False"
#LAMBDA=0.5
#PROJ_MODE="loss_on_projections"
#EPS=0.3
#DEVICE="cpu"
#SEED=0

# === ensemble_regularizer === #
#SCRIPT="ensemble_regularizer"
#DATASET_NAME="mnist"
#TEST="False"
#ENSEMBLE_SIZE=5
#PROJ_MODE="loss_on_projections"
#LAMBDA=0.5
#DEVICE="cpu"

##############
# run script #
##############

## cluster
if [ $HOSTNAME != "zenbook" ] ; then
  cd ~/adversarial_examples/src/
  ##export CUDA_VISIBLE_DEVICES=-1 # GPU
fi

## activate environment
if [ $DEVICE = "cpu" ]; then
  source ~/virtualenvs/venv/bin/activate
elif [ $DEVICE = "gpu" ]; then
  conda activate tensorflow-gpu
fi

## set filenames
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS
FILEPATH="${RESULTS}${TIME}_${DATASET_NAME}_${SCRIPT}"
if [ "$TEST" = "True" ] ; then
  FILEPATH="${FILEPATH}_test"
fi
OUT="${FILEPATH}_out.txt"
CLEAN_OUT="${FILEPATH}_clean.txt"
COMPLEXITY="${FILEPATH}_complexity.txt"

if [ $SCRIPT = "baseline" ]; then
  python3 "baseline_convnet.py" $DATASET_NAME $TEST $DEVICE $SEED > $OUT
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ_LIST $SIZE_PROJ_LIST $PROJ_MODE $ATTACK $EPS $DEVICE>> $OUT
elif [ $SCRIPT = "parallel_randens" ]; then
  python3 "parallel_random_ensemble.py" $DATASET_NAME $TEST 0 $N_PROJ $SIZE_PROJ_LIST $PROJ_MODE $DEVICE>> $OUT
elif [ $SCRIPT = "randreg" ]; then
  python3 "random_regularizer.py" $DATASET_NAME $TEST $LAMBDA $PROJ_MODE $EPS $DEVICE $SEED >> $OUT
elif [ $SCRIPT = "ensemble_regularizer" ]; then
  python3 "ensemble_regularizer.py" $DATASET_NAME $TEST $ENSEMBLE_SIZE $PROJ_MODE $LAMBDA $DEVICE >> $OUT
fi
#!/bin/bash

###########
#  guide  #
###########
##  DATASET_NAME    Supported: mnist,cifar.
##  TEST            If True only takes 100 samples.
##  ATTACK          Supported: None, fgsm, pgd, deepfool, carlini_linf.
##  N_PROJ_LIST     Supported: lists containing 0,..,15. Default for training is [15], default for testing is [6,9,12,15].
##  SIZE_PROJ_LIST  Supported: list containing 8, 12, 16, 20. Default is [8 12 16 20].
##  PROJ_MODE       Supported for randens, parallel_randens: flat, channels, one_channel, grayscale.
##                  Supported for randreg: channels, grayscale.

##########################################
# settings -> deactivate unwanted lines! #
##########################################

# === baseline === #
#SCRIPT="baseline"
#DATASET_NAME="mnist"
#TEST="False"
#ATTACK="fgsm"

# === randens === #
SCRIPT="randens"
DATASET_NAME="mnist"
TEST="False"
N_PROJ_LIST=[6,9,12,15]
SIZE_PROJ_LIST=[8,12,16,20]
PROJ_MODE="channels" # Default: channels
ATTACK=None

# === parallel_randens === #
#SCRIPT="parallel_randens"
#DATASET_NAME="cifar"
#TEST="True"
##N_PROJ=15 # train the maximum n. of projections
#SIZE_PROJ_LIST=[8,12,16,20]
#PROJ_MODE="grayscale"

# === randreg === #
#SCRIPT="randreg"
#DATASET_NAME="mnist"
#TEST="False"
#LAMBDA=0.5
#PROJ_MODE="channels" # Defaults: channels for mnist, grayscale for cifar


##############
# run script #
##############

## cluster
if [ $HOSTNAME != "zenbook" ] ; then
  cd ~/adversarial_examples/src/
  ##export CUDA_VISIBLE_DEVICES=-1 # GPU
fi

source ~/virtualenvs/venv/bin/activate

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
  python3 "baseline_convnet.py" $DATASET_NAME $TEST $ATTACK > $OUT
  sed -n '/ETA:/!p' $OUT > $CLEAN_OUT
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ_LIST $SIZE_PROJ_LIST $PROJ_MODE $ATTACK >> $OUT
  sed -n '/ETA:/!p' $OUT  >> $CLEAN_OUT
elif [ $SCRIPT = "parallel_randens" ]; then
  for proj_idx in $(seq 15); do
    python3 "parallel_randens_training.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ_LIST $PROJ_MODE >> $OUT
    sed -n '/ETA:/!p' $OUT  >> $CLEAN_OUT
    grep -e "Rand" -e "Training time for" $OUT >> $COMPLEXITY
  done
elif [ $SCRIPT = "randreg" ]; then
  python3 "random_regularizer.py" $DATASET_NAME $TEST $LAMBDA $PROJ_MODE >> $OUT
  grep -e "batch" -e "time" -e "accu" -B 8 $OUT  >> $CLEAN_OUT
fi

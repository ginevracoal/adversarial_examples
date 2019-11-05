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
##                  Supported for randreg: no_projections, loss_on_projections, projected_loss, loss_on_perturbations.
## ENSEMBLE_SIZE    Number of models for ensemble_regularizer

##########################################
# settings -> deactivate unwanted lines! #
##########################################

# === baseline === #
#SCRIPT="baseline"
#DATASET_NAME="mnist"
#TEST="False"
#ATTACK="fgsm"
#EPS=0.5

# === randens === #
#SCRIPT="randens"
#DATASET_NAME="mnist"
#TEST="True"
#N_PROJ_LIST=[10] #[6,9,12,15]
#SIZE_PROJ_LIST=[12] #[8,12,16,20]
#PROJ_MODE="channels"
#ATTACK="fgsm"
#EPS=0.3
#DEVICE="gpu"

# === parallel_random_ensemble === #
SCRIPT="parallel_randens"
DATASET_NAME="mnist"
TEST="True"
PROJ_IDX=0
SIZE_PROJ_LIST=[8] #[8,12,16,20]
PROJ_MODE="channels"
DEVICE="gpu"

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
  python3 "baseline_convnet.py" $DATASET_NAME $TEST $ATTACK $EPS > $OUT
  sed -n '/ETA:/!p' $OUT > $CLEAN_OUT
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ_LIST $SIZE_PROJ_LIST $PROJ_MODE $ATTACK $EPS $DEVICE>> $OUT
  sed -n '/ETA:/!p' $OUT  >> $CLEAN_OUT
elif [ $SCRIPT = "parallel_randens" ]; then
#  python3 "parallel_random_ensemble.py" $DATASET_NAME $TEST $PROJ_IDX $SIZE_PROJ_LIST $PROJ_MODE $DEVICE >> $OUT
  python3 "parallel_random_ensemble.py" $DATASET_NAME $TEST 0 $SIZE_PROJ_LIST $PROJ_MODE $DEVICE >> $OUT
  for proj_idx in $(seq 50); do #2); do
    python3 "parallel_random_ensemble.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ_LIST $PROJ_MODE $DEVICE >> $OUT
  done
#  sed -n '/ETA:/!p' $OUT  > $CLEAN_OUT
  grep -e "time" -e "accu" -B 8 $OUT  >> $CLEAN_OUT
  grep -e "Rand" -e "Training time for" $OUT > $COMPLEXITY
elif [ $SCRIPT = "randreg" ]; then
  python3 "random_regularizer.py" $DATASET_NAME $TEST $LAMBDA $PROJ_MODE $EPS $DEVICE $SEED >> $OUT
  grep -e "batch" -e "time" -e "accu" -B 8 $OUT  >> $CLEAN_OUT
elif [ $SCRIPT = "ensemble_regularizer" ]; then
  python3 "ensemble_regularizer.py" $DATASET_NAME $TEST $ENSEMBLE_SIZE $PROJ_MODE $LAMBDA $DEVICE >> $OUT
  grep -e "batch" -e "time" -e "accu" -B 8 $OUT  >> $CLEAN_OUT
fi

## deactivate environment
#if [ $DEVICE = "cpu" ]; then
#  deactivate venv
#elif [ $DEVICE = "gpu" ]; then
#  conda deactivate
#fi
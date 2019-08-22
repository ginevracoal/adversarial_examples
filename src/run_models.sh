#!/bin/bash

############
# settings #
############
SCRIPT="randreg" # "baseline", "randens", "parallel_randens", "randreg"

# === all scripts === #
DATASET_NAME="mnist" # supported: "mnist","cifar"
TEST="False" # if True only takes 100 samples

# === baseline, randens === #
ATTACK=None # supported: "fgsm, "pgd", "deepfool", "carlini_linf"

# === randens === #
N_PROJ_LIST=[6,9,12,15] # Supported: lists containing 0,..,15. Default for training is [15], default for testing is [6,9,12,15]

# === randens, parallel_randens === #
SIZE_PROJ_LIST=[8,12,16,20] # Supported: list containing 8, 12, 16, 20. Default is [8 12 16 20]

# === parallel_randens === #
#N_PROJ=15 # default is 15

# === randreg === #
LAMBDA=0.5

# === clusterino === #
#rm screenlog.0
cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

##############
# run script #
##############
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
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ_LIST $SIZE_PROJ_LIST $ATTACK >> $OUT
  sed -n '/ETA:/!p' $OUT  >> $CLEAN_OUT
elif [ $SCRIPT = "parallel_randens" ]; then
  for proj_idx in $(seq 0 15); do
    python3 "parallel_randens_training.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ_LIST >> $OUT
    sed -n '/ETA:/!p' $OUT  >> $CLEAN_OUT
    grep -e "Rand" -e "Training time for" $OUT >> $COMPLEXITY
  done
elif [ $SCRIPT = "randreg" ]; then
  python3 "random_regularizer.py" $DATASET_NAME $TEST $LAMBDA >> $OUT
  grep -e "batch" -e "time" -e "accu" -B 8 $OUT  >> $CLEAN_OUT
fi

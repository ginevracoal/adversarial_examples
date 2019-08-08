#!/bin/bash

############
# settings #
############
SCRIPT="randens" # "baseline", "randens", "parallel_randens"

# === all scripts === #
DATASET_NAME="mnist" # supported: "mnist","cifar"
TEST="True" # if True only takes 100 samples

# === baseline, randens === #
ATTACK=None # supported: "fgsm, "pgd", "deepfool", "carlini_linf"

# === randens === #
N_PROJ_LIST=[6,9,12,15] # supported: lists containing 0,..,15
		          # default for training is [15], default for testing is [6,9,12,15]
SIZE_PROJ_LIST=[8,12,16,20] # supported: lists containing 8,12,16,20

# === parallel_randens === #
N_PROJ=15 # supported: 0,...,15
SIZE_PROJ=8 # supported: 8, 12, 16, 20

# === clusterino === #
#rm screenlog.0
#cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

##############
# run script #
##############
source ~/virtualenvs/venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
RESULTS="../results/$DATE/"
mkdir -p $RESULTS

OUT="${RESULTS}${DATASET_NAME}_${SCRIPT}"
if [ "$TEST" = "True" ] ; then
    OUT="${OUT}_test"
fi

if [ $SCRIPT = "baseline" ]; then
  BASELINE_OUT="${OUT}_${ATTACK}"
  python3 "baseline_convnet.py" $DATASET_NAME $TEST $ATTACK > "${BASELINE_OUT}_out.txt"
  sed -n '/ETA:/!p' "${BASELINE_OUT}_out.txt" > "${BASELINE_OUT}_clean.txt"
elif [ $SCRIPT = "randens" ]; then
  RANDENS_OUT="${OUT}_${N_PROJ_LIST}_${SIZE_PROJ_LIST}_${ATTACK}"
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ_LIST $SIZE_PROJ_LIST $ATTACK > "${RANDENS_OUT}_out.txt"
  sed -n '/ETA:/!p' "${RANDENS_OUT}_out.txt"  > "${RANDENS_OUT}_clean.txt"
elif [ $SCRIPT = "parallel_randens" ]; then
  for proj_idx in $(seq $N_PROJ); do
    PARALLEL_RANDENS_OUT="${OUT}_${proj_idx}_${SIZE_PROJ}"
    python3 "parallel_randens_training.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ > "${PARALLEL_RANDENS_OUT}_out.txt"
    grep -e "Rand" -e "Training time for" "${PARALLEL_RANDENS_OUT}_out.txt" >> "${OUT}_${SIZE_PROJ}_complexity.txt"
  done
fi

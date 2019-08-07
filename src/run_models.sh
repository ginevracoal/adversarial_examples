#!/bin/bash

############
# settings #
############
SCRIPT="parallel_randens" # "baseline", "randens", "parallel_randens"

# === all scripts === #
DATASET_NAME="mnist" # supported: "mnist","cifar"
TEST="True" # if True only takes 100 samples

# === baseline, randens === #
ATTACK="fgsm" # supported: "fgsm, "pgd", "deepfool", "carlini_linf"

# === randens === #
N_PROJ_LIST=[6]#,9,12,15] # supported: lists containing 0,..,15
		          # default for training is [15], default for testing is [6,9,12,15]
SIZE_PROJ_LIST=[8]#,12,16,20] # supported: lists containing 8,12,16,20

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

OUT_FILENAME="${DATASET_NAME}_${SCRIPT}"
RESULTS="../results"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

source ~/virtualenvs/venv/bin/activate
mkdir -p "$RESULTS/$DATE/"

if [ $SCRIPT = "baseline" ]; then
  python3 "baseline_convnet.py" $DATASET_NAME $TEST $ATTACK > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_out.txt"
  sed -n '/ETA:/!p' "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_out.txt" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_clean.txt"
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" $DATASET_NAME $TEST $N_PROJ $SIZE_PROJ $ATTACK > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_out.txt"
  sed -n '/ETA:/!p' "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_out.txt" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_clean.txt"
elif [ $SCRIPT = "parallel_randens" ]; then
  for proj_idx in $(seq 0 $N_PROJ); do
    python3 "parallel_randens_training.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ > "$RESULTS/$DATE/${OUT_FILENAME}_${proj_idx}_${TIME}_out.txt"
    grep -e "Rand" -e "Training" "$RESULTS/$DATE/${OUT_FILENAME}_${proj_idx}_${TIME}_out.txt" >> "$RESULTS/$DATE/${OUT_FILENAME}_complexity.txt"
  done
fi

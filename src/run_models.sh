#!/bin/bash

############
# settings #
############

SCRIPT="parallel_randens_training" # "baseline", "randens", "parallel_randens_training"

## only for parallel randens
DATASET_NAME="cifar"
TEST="False"
N_PROJ=15
SIZE_PROJ=8 # 8, 12, 16, 20

## only for clusterino
rm screenlog.0
cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

##############
# run script #
##############

OUT_FILENAME=$SCRIPT
RESULTS="../results/"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

source ~/virtualenvs/venv/bin/activate
mkdir -p "$RESULTS/$DATE/"

if [ $SCRIPT = "baseline" ]; then
  python3 "baseline_convnet.py" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt
elif [ $SCRIPT = "parallel_randens_training" ]; then
  for proj_idx in $(seq 11 15); do
    python3 "parallel_randens_training.py" $DATASET_NAME $TEST $proj_idx $SIZE_PROJ > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt
  done
fi

## remove the unnecessary infos from output text
sed -n '/ETA:/!p' "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_clean.txt"

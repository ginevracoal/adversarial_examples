#!/bin/bash

############
# settings #
############

MODEL_NAME="random_ensemble"
RESULTS="../trained_models/"
DATE=$(date +%Y%m%d)
DATETIME=$(date +%Y%m%d%H%M)

## clusterino
#
#export CUDA_VISIBLE_DEVICES=-1 # GPU

## local
source ~/virtualenvs/venv/bin/activate

##############
# run script #
##############

mkdir -p "$RESULTS/$DATE/"
python "${MODEL_NAME}.py" > "$RESULTS/$DATE/${MODEL_NAME}_${DATETIME}"_out.txt

## remove the unnecessary infos from output text
#sed -n '/ETA:/!p' "results/${MODEL_NAME}_out.txt" > "results/${MODEL_NAME}_out_clean.txt"
#!/bin/bash

############
# settings #
############

MODEL_NAME="baseline_convnet"
RESULTS="../results/"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

## clusterino
rm screenlog.0
cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

## local
#source ~/virtualenvs/venv/bin/activate

##############
# run script #
##############

mkdir -p "$RESULTS/$DATE/"
python "${MODEL_NAME}.py" > "$RESULTS/$DATE/${MODEL_NAME}_${TIME}"_out.txt

## remove the unnecessary infos from output text
sed -n '/ETA:/!p' "$RESULTS/$DATE/${MODEL_NAME}_${TIME}"_out.txt > "$RESULTS/$DATE/${MODEL_NAME}_${TIME}_clean.txt"

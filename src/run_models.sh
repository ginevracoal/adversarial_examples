#!/bin/bash

############
# settings #
############

MODEL_NAME="random_ensemble" # this should be the name of the script
FILENAME="" # this should be the name of the output file

## only for clusterino:
rm screenlog.0
cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

##############
# run script #
##############

RESULTS="../results/"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

source ~/virtualenvs/venv/bin/activate
mkdir -p "$RESULTS/$DATE/"
python "${MODEL_NAME}.py" > "$RESULTS/$DATE/${FILENAME}_${TIME}"_out.txt

## remove the unnecessary infos from output text
sed -n '/ETA:/!p' "$RESULTS/$DATE/${FILENAME}_${TIME}"_out.txt > "$RESULTS/$DATE/${FILENAME}_${TIME}_clean.txt"

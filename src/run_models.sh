#!/bin/bash

############
# settings #
############

SCRIPT="baseline" # "baseline", "randens"
OUT_FILENAME="testing_baseline"

## clusterino
#rm screenlog.0
#cd ~/adversarial_examples/src/
#export CUDA_VISIBLE_DEVICES=-1 # GPU

##############
# run script #
##############

RESULTS="../results/"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)

source ~/virtualenvs/venv/bin/activate
mkdir -p "$RESULTS/$DATE/"

if [ $SCRIPT = "baseline" ]; then
  python3 "baseline_convnet.py" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt
elif [ $SCRIPT = "randens" ]; then
  python3 "random_ensemble.py" > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt
fi

## remove the unnecessary infos from output text
sed -n '/ETA:/!p' "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}"_out.txt > "$RESULTS/$DATE/${OUT_FILENAME}_${TIME}_clean.txt"

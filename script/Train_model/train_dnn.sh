#!/bin/bash
mkdir -p ./logs/train/
mkdir -p ./tensorboard_logs

data_type="$1"
hidden_size=20

log_out="./logs/train/${data_type}_hs${hidden_size}.out"
log_err="./logs/train/${data_type}_hs${hidden_size}.err"
echo "▶ DNN TRAINING: data_type=$data_type, hidden_size=$hidden_size"
python ./ensemble_dnn.py --data_type "$data_type" --mode train --hidden_size "$hidden_size" > "$log_out" 2> "$log_err"

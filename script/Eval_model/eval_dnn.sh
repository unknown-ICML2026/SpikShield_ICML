#!/bin/bash
mkdir -p ./logs/eval/
mkdir -p ./tensorboard_logs

data_type="$1"
hidden_size=20

log_out="./logs/eval/${data_type}_hs${hidden_size}_eval.out"
log_err="./logs/eval/${data_type}_hs${hidden_size}_eval.err"
echo "▶ DNN EVALUATION: data_type=$data_type, hidden_size=$hidden_size"
python ./ensemble_dnn.py --data_type "$data_type" --mode eval --hidden_size "$hidden_size" > "$log_out" 2> "$log_err"

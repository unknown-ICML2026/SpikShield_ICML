#!/bin/bash
mkdir -p ./logs/attack/
mkdir -p ./tensorboard_logs

data_type="$1"
hidden_size=20

log_out="./logs/attack/attack_${data_type}_dnn_hs${hidden_size}_eval.out"
log_err="./logs/attack/attack_${data_type}_dnn_hs${hidden_size}_eval.err"
echo "▶ DNN ATTACK EVAL: data_type=$data_type, hidden_size=$hidden_size"
python ./attack_model.py --data_type "$data_type" --mode eval --hidden_size "$hidden_size" --target dnn > "$log_out" 2> "$log_err"

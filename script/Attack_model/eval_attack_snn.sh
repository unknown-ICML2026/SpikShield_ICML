#!/bin/bash
mkdir -p ./logs/attack/
mkdir -p ./tensorboard_logs

data_type="$1"
num_steps=5
hidden_size=20

log_out="./logs/attack/attack_${data_type}_snn_ns${num_steps}_hs${hidden_size}_eval.out"
log_err="./logs/attack/attack_${data_type}_snn_ns${num_steps}_hs${hidden_size}_eval.err"
echo "▶ SNN ATTACK EVAL: data_type=$data_type, num_steps=$num_steps, hidden_size=$hidden_size"
python ./attack_model.py --data_type "$data_type" --mode eval --num_steps "$num_steps" --hidden_size "$hidden_size" --beta 0.9 --target snn > "$log_out" 2> "$log_err"

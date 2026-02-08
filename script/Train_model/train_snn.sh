#!/bin/bash
mkdir -p ./logs/train/
mkdir -p ./tensorboard_logs

data_type="$1"
num_steps=5
hidden_size=20

log_out="./logs/train/${data_type}_ns${num_steps}_hs${hidden_size}.out"
log_err="./logs/train/${data_type}_ns${num_steps}_hs${hidden_size}.err"
echo "▶ TRAINING: data_type=$data_type, num_steps=$num_steps, hidden_size=$hidden_size"
python ./ensemble_snn.py --data_type $data_type --mode train --num_steps $num_steps --hidden_size $hidden_size --beta 0.9 > "$log_out" 2> "$log_err"

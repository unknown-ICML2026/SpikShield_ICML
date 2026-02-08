#!/bin/bash
mkdir -p ./logs/eval/
mkdir -p ./tensorboard_logs

data_type="$1"
num_steps=5
hidden_size=20

log_out="./logs/eval/${data_type}_ns${num_steps}_hs${hidden_size}_eval.out"
log_err="./logs/eval/${data_type}_ns${num_steps}_hs${hidden_size}_eval.err"
echo "▶ EVALUATION: data_type=$data_type, num_steps=$num_steps, hidden_size=$hidden_size"
python ./ensemble_snn.py --data_type $data_type --mode eval --num_steps $num_steps --hidden_size $hidden_size --beta 0.9 > "$log_out" 2> "$log_err"

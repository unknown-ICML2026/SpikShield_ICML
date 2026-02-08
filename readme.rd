SpikShield_ICML Usage

Prerequisites
- Prepare datasets under ./dataset/{bcr,mnist,ecg} with the filenames used in code.
- Checkpoints are written under ./checkpoints and ./checkpoints_attack.
- Attack caches are written under ./attack_datasets.

1. Train ensemble models
   1. SNN
      - bash script/Train_model/train_snn.sh bcr
      - bash script/Train_model/train_snn.sh mnist
      - bash script/Train_model/train_snn.sh ecg
   2. DNN
      - bash script/Train_model/train_dnn.sh bcr
      - bash script/Train_model/train_dnn.sh mnist
      - bash script/Train_model/train_dnn.sh ecg

2. Train attack models
   1. SNN target
      - bash script/Attack_model/train_attack_snn.sh bcr
      - bash script/Attack_model/train_attack_snn.sh mnist
      - bash script/Attack_model/train_attack_snn.sh ecg
   2. DNN target
      - bash script/Attack_model/train_attack_dnn.sh bcr
      - bash script/Attack_model/train_attack_dnn.sh mnist
      - bash script/Attack_model/train_attack_dnn.sh ecg

3. Evaluate attack models
   1. SNN target
      - bash script/Attack_model/eval_attack_snn.sh bcr
      - bash script/Attack_model/eval_attack_snn.sh mnist
      - bash script/Attack_model/eval_attack_snn.sh ecg
   2. DNN target
      - bash script/Attack_model/eval_attack_dnn.sh bcr
      - bash script/Attack_model/eval_attack_dnn.sh mnist
      - bash script/Attack_model/eval_attack_dnn.sh ecg

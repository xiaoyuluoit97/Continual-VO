# Continual Visual Odometry
This directory contains code to reproduce the results in our paper:

"The Empirical Impact of Forgetting and Transfer in Continual Visual Odometry"
https://arxiv.org/pdf/2406.01797
by Paolo Cudrano*, Xiaoyu Luo*, and Matteo Matteucci. CoLLAs 2024

### Environment Setup
Please check our docker hub:
https://hub.docker.com/repository/docker/luoxiaoyuwow/cl-baseline/general

You can execute :
```
docker pull luoxiaoyuwow/cl-baseline:latest 
```

### Training Setup
Please check the config.json:
```
  "TIMES": "0", # The resume times
  "TRAIN": "replay_full", # The experiments name
  "RESUME_PATH": "log/replay_full", # The resume path
  "RESUME_FILE": "replay_fullExp34_resume70time.pth", # The resume 
  "STARTEXP": 70, # Resume start from experience number
  "LOAD_FROM_NUMBERONE": "/Continual-VO/log/naive_pth/naive_Exp0_FINAL.pth", # .pth file path
  "TEST": false, # testing flag
  "ESUME_TRAINR": false, # resume training
  "NUMOFTRAINING": 27776, # 13888 normal training, 27776 is full buffer size training
  "dataset_path": "/dataset/vo_dataset/test-buffer" # please check the path
```
To train & test the continual learning VO baseline 
```
python continual_vo.py
```
To train & test the continual learning VO with action embedding

```
python continual_vo_act.py
```
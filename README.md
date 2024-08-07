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
### Dataset generation & Joint training baseline reproduce
Our joint training baseline and dataset generation are based on the work of Zhao et al. Please check their git repo for setup. 

https://github.com/Xiaoming-Zhao/PointNav-VO

To generate the dataset scene by scene, please use the following .py file in our repo.
```
vo/dataset/generate_datasets_continual.py

vo/models/vo_cnn.py
```
To reproduce the joint baseline, please use the following model
```
@baseline_registry.register_vo_model(name="vo_cnn_rgb_d")

@baseline_registry.register_vo_model(name="vo_cnn_rgb_d_emb")
```
After generating please run the .sh file to shuffle all trajectories. (Don't do it if you would like to simulate a complete sequence scenario)
```
dataset_preparation/shuffle.sh
```
Please check the readme in /dataset_preparation for more details about the replay dataset setup

### Training Setup
Please check the config.json:
```
  "TIMES": "0", # The resume times
  "TRAIN": "replay_full", # The experiments name
  "RESUME_PATH": "log/replay_full", # The resume dir path
  "RESUME_FILE": "replay_fullExp34_resume70time.pth", # Which consist by "(experiemnt name)_(experience numbers)_(resume times).pth"
  "STARTEXP": 70, # Resume start from experience number
  
  "TEST": false, # testing flag
  "ESUME_TRAINR": false, # resume training
  "NUMOFTRAINING": 13888, # 13888 normal training, 27776 is full buffer size training
  "dataset_path": "/dataset/vo_dataset/test-buffer" # path of dataset
  
  "USE_EWC": false,     # use the EWC
  "EWC_LAMBDA": 0.25,
  "USE_LWF": false,     # use the LwF
  "LWF_ALPHA": 0.5,
  "LWF_TEMP": 0.1,
```

To train & test the continual learning VO baseline 
```
python continual_vo.py
```
To train & test the continual learning VO with action embedding

```
python continual_vo_act.py
```

### Result Visualization
We use wandb to monitor the training process and visualize the training results. Please check here to deploy your own wandb.

https://wandb.ai/

### Contact
Please feel free to contact xiaoyu.luo.it@gmail.com for any reproducing questions.



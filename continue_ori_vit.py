import gc
import os
import random
from avalanche.evaluation import PluginMetric
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
import re
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, forgetting_metrics, bwt_metrics,\
    confusion_matrix_metrics, cpu_usage_metrics, \
    disk_usage_metrics, gpu_usage_metrics, MAC_metrics, \
    ram_usage_metrics, timing_metrics
from loss_function_avalanche import predict_diff_loss
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.training import OnlineNaive
from datetime import datetime
import sys
import wandb
#from memory_profiler import profile
from asy_loaded_actemb import avl_data_set
from avalanche.training.supervised import Naive,EWC,LwF
from avalanche.training.templates import SupervisedTemplate
#from kubernetesdlprofile import kubeprofiler
from avalanche.evaluation.metrics import (
    timing_metrics,
    loss_metrics
)
from avalanche.training.plugins import EvaluationPlugin,LwFPlugin,EWCPlugin
import vision_transformer

import early
from torchvision.transforms import ToTensor

from resnet_backbone import VisualOdometryCNNBase
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from utils.misc_utils import Flatten
from utils.baseline_registry import baseline_registry
from model_utils.visual_encoders import resnet
from model_utils.running_mean_and_var import RunningMeanAndVar
from vo.common.common_vars import *
from custom_save_plugin import CustomSavePlugin
from avalanche.logging import (
    InteractiveLogger,
    TextLogger,
    CSVLogger,
    TensorboardLogger,
)
from typing import List, Dict
#key of wandb a19e31fa13d7342a558bd4041f695ce47c85cb4f
from avalanche.benchmarks.scenarios.generic_scenario import CLExperience

RGB_PAIR_CHANNEL = 6
DEPTH_PAIR_CHANNEL = 2
DELTA_DIM = 3

TRAIN = "naive_vit_2head"
RESUME_PATH = "log/naive_vit_2head"
TIMES="3"
RESUME_FILE = "act_emb_naive_vitExp21_resume2time.pth"
DATA_FOLDER_PATH = "/custom/dataset/vo_dataset/test-72exp"
OBSERVATION_SPACE = ["rgb", "depth"]
ESUME_TRAINR = False
NORMALIZE = False
DEVICE = "cuda:7"
VOTRAIN_LR = 2.5e-4
VOTRAIN_ESP = 1.0e-8
VOTRAIN_WEIGHT_DECAY = 0.0
OBSERVATION_SIZE = (
    341,
    192,
)
ACTION_EMBEDDING = True
def optimizer_continue(model,name):
    if name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=VOTRAIN_LR,
            #eps=VOTRAIN_ESP,
            weight_decay=VOTRAIN_WEIGHT_DECAY
        )
    elif name =="Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=VOTRAIN_LR,
            eps=VOTRAIN_ESP,
            weight_decay=VOTRAIN_WEIGHT_DECAY
        )

    return optimizer

def model_select(name,action_embedding,normalize):
    if name == 'VIT':
        model = vision_transformer.vit_small_patch16_224(
            pretrained=False,
            img_size=(320, 160),
            in_chans=3,
            splite_head=True,
            class_token=action_embedding,
            normalize_visual_inputs=normalize,
            global_pool='avg',
            num_classes=3
        )
    elif name == "resnet50":
        model = VisualOdometryCNNBase(
            observation_space=OBSERVATION_SPACE,
            observation_size=OBSERVATION_SIZE,
            action_embedding=action_embedding,
            normalize_visual_inputs=normalize,
            rgb_pair_channel=RGB_PAIR_CHANNEL,
            depth_pair_channel=DEPTH_PAIR_CHANNEL,
            output_dim=DELTA_DIM,
            backbone="resnet50",
            resnet_baseplanes=64)

    elif name == "resnet18":
        model = VisualOdometryCNNBase(
            observation_space=OBSERVATION_SPACE,
            observation_size=OBSERVATION_SIZE,
            action_embedding=action_embedding,
            normalize_visual_inputs=normalize,
            rgb_pair_channel=RGB_PAIR_CHANNEL,
            depth_pair_channel=DEPTH_PAIR_CHANNEL,
            output_dim=DELTA_DIM,
            backbone="resnet18",
            resnet_baseplanes=32)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Now use model:{name} , the total paramter is: {total_params}")
    return model

def load_savepoint(name,actionembedding,model,resume_path,resume_file):
    match = re.match(r".*Exp(\d+)_", resume_file)
    if match:
        exp_number = int(match.group(1))
        initexperience = exp_number + 1
        print(f"now start from experience {initexperience}")
    else:
        print("not find any .pth files")
        initexperience = -1

    if name == 'VIT' and actionembedding:
        checkpoint = torch.load(os.path.join(resume_path,resume_file)) # 加载检查点
        cls_token_shape = checkpoint['cls_token'].shape
        new_cls_token = torch.randn(cls_token_shape)
        model.cls_token = torch.nn.Parameter(new_cls_token)
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(os.path.join(resume_path,resume_file)))


    return initexperience,model

def main():
    #deterministic setting
    seed_value = 10777140
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    device = torch.device(DEVICE)
    model = model_select("resnet18",ACTION_EMBEDDING,NORMALIZE)

    initexperience = 0
    if ESUME_TRAINR:
        initexperience, model = load_savepoint("resnet18",ACTION_EMBEDDING, model, RESUME_PATH, RESUME_FILE)
    optimizer = optimizer_continue(model.to(device), "Adam")
    criterion = predict_diff_loss()

    #wandb configue
    #pjn = "VIT-Training-dataset"
    pjn = 'cleantest'
    wb_logger=WandBLogger(
         project_name=pjn,       # set the wandb project where this run will be logged
         # track hyperparameters and run metadata
         config={
             "learning_rate": VOTRAIN_LR,
             "Nor": NORMALIZE,
             "action_emb": False,
             "architecture": "CNN-Resnet18-TwoHiddenlayer512",
             "dataset": "Habitat-Gibson-V2",
             "epochs": 60,
         })
    custom_plugin = CustomSavePlugin(TIMES,TRAIN,RESUME_PATH)

    # print to stdout
    interactive_logger = InteractiveLogger()
    # build path
    params_identifier = f"{TRAIN}resume{TIMES}_{ESUME_TRAINR}"
    # datetime stamp
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # build path
    log_folder = f"{RESUME_PATH}/{current_datetime}_{params_identifier}/"
    # pass path to logger
    csv_logger = CSVLogger(log_folder=log_folder)
    text_logger = TextLogger(open(f"{log_folder}log.txt", "a"))


    eval_plugin = EvaluationPlugin(
        #EWCPlugin(ewc_lambda=0.3),
        #LwFPlugin(alpha=0.5, temperature=0.1),
        custom_plugin,
        early.EarlyStoppingPlugin(patience=8,val_stream_name="test_stream",metric_name="Loss_Stream",mode="min",peval_mode="epoch",margin=0.0,verbose=True),
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        #,wb_logger
        loggers=[interactive_logger, text_logger, csv_logger],
        collect_all=True,
    )

    cl_strategy = Naive(
        model=model,
        train_mb_size=32,
        optimizer=optimizer,
        criterion=criterion,
        evaluator=eval_plugin,
        train_epochs=60,
        device=device,
        eval_every=1
    )

    benchmark,test_benchmark= avl_data_set(ACTION_EMBEDDING,DATA_FOLDER_PATH,device)
    training(benchmark,test_benchmark,cl_strategy,initexperience)

#@profile(precision=2,stream=open('memorytest/training.log','w+'))
def training(benchmark,cl_strategy,initial_exp):

    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience: ", experience.current_experience)
        print("Train dataset contains", len(experience.dataset), "instances")
        i = experience.current_experience
        cl_strategy.train(experience,eval_streams=[benchmark.test_stream[i:(i+1)]],shuffle=False)
        gc.collect()
        print("Training & validation completed,test starting")
        #cl_strategy.eval(teststream_from_benchmark.test_stream,shuffle=False)
        print("Evaluation completed")
        #break

if __name__ == "__main__":
    main()
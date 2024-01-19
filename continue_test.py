import gc
import random
from avalanche.evaluation import PluginMetric
import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, forgetting_metrics, bwt_metrics,\
    confusion_matrix_metrics, cpu_usage_metrics, \
    disk_usage_metrics, gpu_usage_metrics, MAC_metrics, \
    ram_usage_metrics, timing_metrics
from loss_function_avalanche import predict_diff_loss
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training import OnlineNaive
from avalanche.training.supervised.strategy_wrappers_online import OnlineNaive
import sys
import wandb

from asy_loaded_avldataset import avl_data_set
from avalanche.training.supervised import Naive,EWC,LwF
from avalanche.training.templates import SupervisedTemplate
#from kubernetesdlprofile import kubeprofiler
from avalanche.evaluation.metrics import (
    timing_metrics,
    loss_metrics
)
from avalanche.training.plugins import LRSchedulerPlugin, EvaluationPlugin

import early

RGB_PAIR_CHANNEL = 6
DEPTH_PAIR_CHANNEL = 2
DELTA_DIM = 3
ESUME_TRAINR = False

RESUME_STATE_FILE = "/home/*******************"
OBSERVATION_SPACE = ["rgb", "depth"]

VOTRAIN_LR = 2.0e-4

VOTRAIN_ESP = 1.0e-8
VOTRAIN_WEIGHT_DECAY = 0.0
from torchvision.transforms import ToTensor
OBSERVATION_SIZE = (
    341,
    192,
)
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from utils.misc_utils import Flatten
from utils.baseline_registry import baseline_registry
from model_utils.visual_encoders import resnet
from model_utils.running_mean_and_var import RunningMeanAndVar
from vo.common.common_vars import *

from avalanche.logging import (
    InteractiveLogger,
    TextLogger,
    CSVLogger,
    TensorboardLogger,
)
from typing import List, Dict
#key of wandb a19e31fa13d7342a558bd4041f695ce47c85cb4f


class CustomSavePlugin(PluginMetric):
    def __init__(self):
        """
        Creates an instance of a plugin metric.

        Child classes can safely invoke this (super) constructor as the first
        experience.
        """
        self.epoch_num = 0
        self.current_epoch_num = 0
        self.current_iteration_num = 1
        self.current_exp_num = 0
        self.batch_size = 128

        self.currentexp_train_epoch_iter_info = {}  # 用于存储已读取的数据

        self.currentexp_eval_iter_info = []
        self.whole_size = 13888


    def wandblogger(self,loss_result, eval_flag, scence_num,epochs):
        if eval_flag == 1:
            wandb.log({f"test_loss/experience_loss": loss_result[0]})
            wandb.log({f"test/experience_abs_dx": loss_result[1]})
            wandb.log({f"test/experience_abs_dz": loss_result[2]})
            wandb.log({f"test/experience_abs_dyaw": loss_result[3]})
        elif eval_flag == 0:
            wandb.log({f"train_loss/all_general_loss": loss_result[0]})
            wandb.log({f"train/experience_general_loss": loss_result[0]})
            wandb.log({f"train/experience_abs_dx": loss_result[1]})
            wandb.log({f"train/experience_abs_dz": loss_result[2]})
            wandb.log({f"train/experience_abs_dyaw": loss_result[3]})
        elif eval_flag == -1:
            wandb.log({f"validation_loss/experience_loss_by_epochs": loss_result[0]})
            wandb.log({f"validation/experience_abs_dx": loss_result[1]})
            wandb.log({f"validation/experience_abs_dz": loss_result[2]})
            wandb.log({f"validation/experience_abs_dyaw": loss_result[3]})

    def compute_abs_diff(
            self,
            pred_deltax,
            pred_deltaz,
            pred_deltayaw,
            gt_deltax,
            gt_deltaz,
            gt_deltayaw,
    ):
        ##here is the target

        # NOTE: we should not use sqrt in the loss
        # since it may cause NaN in the backward
        assert pred_deltax.size() == gt_deltax.size()
        delta_x_diffs = abs(gt_deltax - pred_deltax)


        assert pred_deltaz.size() == gt_deltaz.size()
        delta_z_diffs = abs((gt_deltaz - pred_deltaz))


        # return loss_dz, abs_diff_dz, target_magnitude_dz, relative_diff_dz
        assert pred_deltayaw.size() == gt_deltayaw.size()
        delta_yaw_diffs = abs(gt_deltayaw - pred_deltayaw)

        return torch.mean(delta_x_diffs) , torch.mean(delta_z_diffs), torch.mean(delta_yaw_diffs)


    def result(self, **kwargs) :
        pass
    def reset(self, **kwargs) -> None:
        pass

    def before_training_epoch(
        self, strategy: "SupervisedTemplate"
    ):
        self.currentexp_train_epoch_iter_info[self.current_epoch_num]=[]

    def after_training_iteration(
        self, strategy: "SupervisedTemplate"
    ):
        abs_dx,abs_dz,abs_dyaw = self.compute_abs_diff(strategy.mb_output[:, 0].unsqueeze(1),
                                strategy.mb_output[:, 1].unsqueeze(1),
                                strategy.mb_output[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 1].unsqueeze(1),
                                strategy.mb_y[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 3].unsqueeze(1))
        #print("now is scencec"+str(strategy.mb_y[:,6].unsqueeze(1)[0]))
        self.currentexp_train_epoch_iter_info[self.current_epoch_num].append([strategy.loss.item(), abs_dx.item(), abs_dz.item(), abs_dyaw.item()])

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """

        result_loss = np.array(self.currentexp_train_epoch_iter_info[self.current_epoch_num])
        result_loss_epoch = np.mean(result_loss, axis=0)
        self.wandblogger(result_loss_epoch,0,strategy.experience.current_experience,self.current_epoch_num)
        self.current_epoch_num = self.current_epoch_num + 1

        torch.save(strategy.model.state_dict(),"experience_weights_latest.pth")
        torch.cuda.synchronize()
        pass


    def after_training_exp(self, strategy: 'PluggableStrategy'):
        del self.currentexp_train_epoch_iter_info
        gc.collect()
        self.currentexp_train_epoch_iter_info = {}
        self.current_epoch_num = 0


    def after_eval_iteration(self, strategy: 'PluggableStrategy'):
        abs_dx,abs_dz,abs_dyaw = self.compute_abs_diff(strategy.mb_output[:, 0].unsqueeze(1),
                                strategy.mb_output[:, 1].unsqueeze(1),
                                strategy.mb_output[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 1].unsqueeze(1),
                                strategy.mb_y[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 3].unsqueeze(1))
        self.currentexp_eval_iter_info.append([strategy.loss.item(), abs_dx.item(), abs_dz.item(), abs_dyaw.item()])

    def after_eval_exp(self, strategy: 'PluggableStrategy'):
        result_loss = np.array(self.currentexp_eval_iter_info)
        result_loss = np.mean(result_loss, axis=0)
        #print("当前准备记录validation,epoch是 ",self.current_epoch_num)
        if self.current_epoch_num == 0:
             self.wandblogger(result_loss, 1, strategy.experience.current_experience,0)
        if self.current_epoch_num != 0:
             self.wandblogger(result_loss, -1, strategy.experience.current_experience,0)
        self.wandblogger(result_loss, 1, strategy.experience.current_experience, 0)
        del self.currentexp_eval_iter_info
        gc.collect()
        self.currentexp_eval_iter_info = []



from avalanche.benchmarks.scenarios.generic_scenario import CLExperience
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        observation_space=OBSERVATION_SPACE,
        observation_size=OBSERVATION_SIZE,
        baseplanes=32,
        ngroups=32,
        spatial_size_w=128,
        spatial_size_h=128,
        make_backbone=None,
        normalize_visual_inputs=False,
        after_compression_flat_size=2048,
        rgb_pair_channel=RGB_PAIR_CHANNEL,
        depth_pair_channel=DEPTH_PAIR_CHANNEL,
        discretized_depth_channels=0,
        top_down_view_pair_channel=False,
    ):
        super().__init__()
        ##set up rgb, 6 per pair (3*2)
        if "rgb" in observation_space:
            self._n_input_rgb = rgb_pair_channel
            spatial_size_w, spatial_size_h = observation_size
        else:
            self._n_input_rgb = 0
        ##set up depth
        if "depth" in observation_space:
            self._n_input_depth = depth_pair_channel
            spatial_size_w, spatial_size_h = observation_size
        else:
            self._n_input_depth = 0

        ##input channels defination here
        input_channels = (
            self._n_input_depth
            + self._n_input_depth
            + self._n_input_rgb
        )

        # NOTE: visual odometry must not be blind
        assert input_channels > 0

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(input_channels)
        else:
            self.running_mean_and_var = nn.Sequential()

        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        final_spatial_w = int(
            np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
        )
        final_spatial_h = int(
            np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
        )
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial_w * final_spatial_h))
        )

        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial_h,
            final_spatial_w,
        )

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observation_pairs):

        cnn_input = []

        if self._n_input_rgb > 0:
            rgb_observations = observation_pairs[:, :, :, :6]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            # [prev_rgb, cur_rgb]
            cnn_input.append(
                [
                    rgb_observations[:, : self._n_input_rgb // 2, :],
                    rgb_observations[:, self._n_input_rgb // 2 :, :],
                ]
            )

        act_observations = observation_pairs[:, :, :, -1:]
        act_observations = act_observations.permute(0, 3, 1, 2)
        cnn_input.append(
            [
                act_observations,
                act_observations,
            ]
        )
        if self._n_input_depth > 0:
            # depth_observations = observation_pairs[:, :, :, -2:]
            depth_observations = observation_pairs[:, :, :, -3:-1]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(
                [
                    depth_observations[:, : self._n_input_depth // 2, :],
                    depth_observations[:, self._n_input_depth // 2 :, :],
                ]
            )

        print()
        # input order:
        #[0 is pre_rgb; 1 is prev_depth ; 2 is cur_rgb ; 3 cur_depth]
        # [prev_rgb, prev_depth, prev_discretized_depth, prev_top_down_view,
        #  cur_rgb, cur_depth, cur_discretized_depth, cur_top_down_view]
        cnn_input = [j for i in list(zip(*cnn_input)) for j in i]

        x = torch.cat(cnn_input, dim=1)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x

class VisualOdometryCNNBase(nn.Module):
    def __init__(
        self,
        *,
        observation_space=OBSERVATION_SPACE,
        observation_size=OBSERVATION_SIZE,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        normalize_visual_inputs=False,
        output_dim=DELTA_DIM,
        dropout_p=0.2,
        after_compression_flat_size=2048,
        rgb_pair_channel=RGB_PAIR_CHANNEL,
        depth_pair_channel=DEPTH_PAIR_CHANNEL,
        discretized_depth_channels=0,
        top_down_view_pair_channel=0,
    ):
        super().__init__()

        self.visual_encoder = ResNetEncoder(
            observation_space=observation_space,
            observation_size=observation_size,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            after_compression_flat_size=after_compression_flat_size,
            rgb_pair_channel=rgb_pair_channel,
            depth_pair_channel=depth_pair_channel,
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_pair_channel=top_down_view_pair_channel,
        )

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(np.prod(self.visual_encoder.output_shape), hidden_size),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs):
        visual_feats = self.visual_encoder(observation_pairs)
        visual_feats = self.visual_fc(visual_feats)
        output = self.output_head(visual_feats)
        #output_dim dx dz dy? dyaw
        ##output is a tensor(128,4)
        return output


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

def main():
    #prof = kubeprofiler.KubeProfiler()
    device = torch.device("cuda")


    model = VisualOdometryCNNBase()
    optimizer = optimizer_continue(model.to(device), "Adam")
    criterion = predict_diff_loss()
    n_classes = 2

    if ESUME_TRAINR:
        resume_ckpt = torch.load(RESUME_STATE_FILE)
        model = load_state_dict(resume_ckpt['model_state_dict'])
        optimizer = load_state_dict(resume_ckpt['optimizer_state_dict'])

    seed_value = 10777140

    # 设置Python的随机种子
    random.seed(seed_value)

    # 设置NumPy的随机种子
    np.random.seed(seed_value)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pjn = "FULL"
    if ESUME_TRAINR:
        pjn="CL-VO-(naive,resume)"
    wb_logger=WandBLogger(
         project_name=pjn,       # set the wandb project where this run will be logged
         # track hyperparameters and run metadata
         config={
             "learning_rate": VOTRAIN_LR,
             "architecture": "CNN-Resnet18-TwoHiddenlayer512",
             "dataset": "Habitat-Gibson-V2",
             "epochs": 30,
         })
    custom_plugin = CustomSavePlugin()

    # log to text file
    text_logger = TextLogger(open("log.txt", "a"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    csv_logger = CSVLogger()
    storage_policy = ReservoirSamplingBuffer(max_size=64)
    replay_plugin = ReplayPlugin(
        mem_size=64, batch_size=1, storage_policy=storage_policy
    )
    eval_plugin = EvaluationPlugin(
        custom_plugin,
        replay_plugin,
        # early.EarlyStoppingPlugin(patience=2,val_stream_name="test_stream",metric_name="Loss_Stream",mode="min",peval_mode="epoch",margin=0.0,verbose=True),
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


    cl_strategy = OnlineNaive(
       model=model,
       train_mb_size=1,
       optimizer=optimizer,
       criterion=criterion,
       train_passes=1,
       eval_mb_size=32,
       evaluator=eval_plugin,
       device=device,
    )

    #cl_strategy = LwF(
    #    model=model,
    #    alpha=0.1,
    #    temperature=0.1,
    #    train_mb_size=128,
    #    optimizer=optimizer,
    #    criterion=criterion,
    #    evaluator=eval_plugin,
    #    train_epochs=40,
    #    device=device,
    #)
    #cl_strategy = EWC(
    #    model=model,
    #    train_mb_size=128,
    #    ewc_lambda = 1,
    #    optimizer=optimizer,
    #    criterion=criterion,
    #    evaluator=eval_plugin,
    #    train_epochs=40,
    #    device=device,
    #)
    print("Starting experiment...")
    results = []

    benchmark,validation_benchmark= avl_data_set(device)

    batch_streams = benchmark.streams.values()
    # ocl_benchmark = OnlineCLScenario(batch_streams)
    for i, exp in enumerate(benchmark.train_stream):
        # Create online scenario from experience exp
        ocl_benchmark = OnlineCLScenario(
            original_streams=batch_streams,
            experiences=exp,
            experience_size=1,
            access_task_boundaries=True,
            shuffle=False,
        )
        # Train on the online train stream of the scenario
        cl_strategy.train(ocl_benchmark.train_stream,shuffle=False)
        results.append(cl_strategy.eval(ocl_benchmark.original_test_stream))

    # training(benchmark,validation_benchmark,cl_strategy,0,"prof")





def training(benchmark,validation_benchmark,cl_strategy,initial_exp,prof):

    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience: ", experience.current_experience)
        print("Train dataset contains", len(experience.dataset), "instances")
        i = experience.current_experience
        cl_strategy.train(experience,shuffle=False)
        print("Training & validation completed,test starting")
        #cl_strategy.eval(teststream_from_benchmark.test_stream,shuffle=False)
        #cl_strategy.eval(validation_benchmark.test_stream,shuffle=False)
        print("Evaluation completed")

if __name__ == "__main__":
    main()
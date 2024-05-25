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
from avalanche.training.supervised import Naive,EWC,LwF
from avalanche.training.templates import SupervisedTemplate
#from kubernetesdlprofile import kubeprofiler
from avalanche.evaluation.metrics import (
    timing_metrics,
    loss_metrics
)
from avalanche.training.plugins import EvaluationPlugin,LwFPlugin,EWCPlugin

import early

RGB_PAIR_CHANNEL = 6
DEPTH_PAIR_CHANNEL = 2
DELTA_DIM = 3
TIMES = 999
TRAIN = "FWT"
RESUME_PATH = "log/naive-fwt"
OBSERVATION_SPACE = ["rgb", "depth"]

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


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        action_embedding,
        normalize_visual_inputs,

        baseplanes=32,
        ngroups=32,
        spatial_size_w=128,
        spatial_size_h=128,
        make_backbone=None,

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

        self.action_embedding=action_embedding
        ##input channels defination here
        if self.action_embedding:
            input_channels = (
                self._n_input_depth
                + 1
                + self._n_input_rgb
            )
        else:
            input_channels = (
                self._n_input_depth
                + self._n_input_rgb
            )
        input_channels = 1
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
        if self.action_embedding:
            if self._n_input_rgb > 0:
                rgb_observations = observation_pairs[:, :, :, :6]
                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                rgb_observations = rgb_observations.permute(0, 3, 1, 2)
                rgb_observations = rgb_observations / 255.0  # normalize RGB
                # [prev_rgb, cur_rgb]
                cnn_input.append(
                    [
                        rgb_observations[:, : self._n_input_rgb // 2, :],
                        rgb_observations[:, self._n_input_rgb // 2:, :],
                    ]
                )

            if self._n_input_depth > 0:
                depth_observations = observation_pairs[:, :, :, -3:-1]
                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                depth_observations = depth_observations.permute(0, 3, 1, 2)
                cnn_input.append(
                    [
                        depth_observations[:, : self._n_input_depth // 2, :],
                        depth_observations[:, self._n_input_depth // 2:, :],
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
            # input order:
            # [0 is pre_rgb; 1 is prev_depth ; 2 is cur_rgb ; 3 cur_depth]
            # [prev_rgb, prev_depth, prev_discretized_depth, prev_top_down_view,
            #  cur_rgb, cur_depth, cur_discretized_depth, cur_top_down_view]
            cnn_input = [j for i in list(zip(*cnn_input)) for j in i][5]
            #cnn_input = [j for i in list(zip(*cnn_input)) for j in i]
        else:
            if self._n_input_rgb > 0:
                rgb_observations = observation_pairs[:, :, :, :6]
                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                rgb_observations = rgb_observations.permute(0, 3, 1, 2)
                rgb_observations = rgb_observations / 255.0  # normalize RGB
                # [prev_rgb, cur_rgb]
                cnn_input.append(
                    [
                        rgb_observations[:, : self._n_input_rgb // 2, :],
                        rgb_observations[:, self._n_input_rgb // 2:, :],
                    ]
                )

            if self._n_input_depth > 0:
                depth_observations = observation_pairs[:, :, :, -2:]
                # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
                depth_observations = depth_observations.permute(0, 3, 1, 2)
                cnn_input.append(
                    [
                        depth_observations[:, : self._n_input_depth // 2, :],
                        depth_observations[:, self._n_input_depth // 2:, :],
                    ]
                )
            # input order:
            # [0 is pre_rgb; 1 is prev_depth ; 2 is cur_rgb ; 3 cur_depth]
            # [prev_rgb, prev_depth, prev_discretized_depth, prev_top_down_view,
            #  cur_rgb, cur_depth, cur_discretized_depth, cur_top_down_view]
            cnn_input = [j for i in list(zip(*cnn_input)) for j in i]

        x = cnn_input
        #x = torch.cat(cnn_input, dim=1)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        del cnn_input
        return x

class VisualOdometryCNNBase(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        normalize_visual_inputs,
        action_embedding,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet18",
        output_dim=DELTA_DIM,
        dropout_p=0,
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
            action_embedding=action_embedding,
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

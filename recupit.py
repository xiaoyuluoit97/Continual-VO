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
#from asy_loaded_avldataset import avl_data_set
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

RGB_PAIR_CHANNEL = 6
DEPTH_PAIR_CHANNEL = 2
DELTA_DIM = 3

TRAIN = "act_emb_naive_vit_final"
RESUME_PATH = "log/act_emb_naive_vit_final"
TIMES="1"
RESUME_FILE = "naive_60epExp70_resume1time.pth"
DATA_FOLDER_PATH = "/custom/dataset/vo_dataset/test-72exp"
OBSERVATION_SPACE = ["rgb", "depth"]

ESUME_TRAINR = True


NORMALIZE = False
DEVICE = "cuda:7"
VOTRAIN_LR = 2.5e-4
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

    def __init__(self, log_folder=None,cur_exp=None):
        """Creates an instance of `CSVLogger` class.
        :param log_folder: folder in which to create log files.
            If None, `csvlogs` folder in the default current directory
            will be used.
        """

        super().__init__()
        self.epoch_num = 0
        self.current_epoch_num = 0
        self.current_iteration_num = 1
        self.current_exp_num = 0
        self.batch_size = 128

        self.currentexp_train_epoch_iter_info = {}  # 用于存储已读取的数据

        self.currentexp_eval_iter_info = []
        self.whole_size = 13888
        self.cur_exp = cur_exp

        self.log_folder = log_folder if log_folder is not None else "csvlogs"
        print("NOW MAKE DIR")

        os.makedirs(self.log_folder, exist_ok=True)
        overall_forward_pred = f"overall_forward_pred_{self.cur_exp}.csv"
        overall_left_pred = f"overall_left_pred_{self.cur_exp}.csv"
        overall_right_pred = f"overall_right_pred_{self.cur_exp}.csv"

        self.overall_forward_pred = open(
            os.path.join(self.log_folder, overall_forward_pred), "w"
        )
        self.overall_left_pred = open(
            os.path.join(self.log_folder, overall_left_pred), "w"
        )
        self.overall_right_pred = open(
            os.path.join(self.log_folder, overall_right_pred), "w"
        )

        perexp_uni_absdelta = f"perexp_all_absdelta_{self.cur_exp}.csv"
        perexp_forward_absdelta = f"perexp_forward_absdelta_{self.cur_exp}.csv"
        perexp_left_absdelta = f"perexp_left_absdelta_{self.cur_exp}.csv"
        perexp_right_absdelta = f"perexp_right_absdelta_{self.cur_exp}.csv"

        self.perexp_uni_absdelta = open(
            os.path.join(self.log_folder, perexp_uni_absdelta), "w"
        )
        self.perexp_forward_absdelta = open(
            os.path.join(self.log_folder, perexp_forward_absdelta), "w"
        )
        self.perexp_left_absdelta = open(
            os.path.join(self.log_folder, perexp_left_absdelta), "w"
        )
        self.perexp_right_absdelta = open(
            os.path.join(self.log_folder, perexp_right_absdelta), "w"
        )


        os.makedirs(self.log_folder, exist_ok=True)

        print(
            "eval_exp",
            "cur_exp",
            "action",
            "predx",
            "predyaw",
            "predz",
            sep=",",
            file=self.overall_forward_pred,
            flush=True,
        )
        print(
            "eval_exp",
            "cur_exp",
            "action",
            "predx",
            "predyaw",
            "predz",
            sep=",",
            file=self.overall_left_pred,
            flush=True,
        )
        print(
            "eval_exp",
            "cur_exp",
            "action",
            "predx",
            "predyaw",
            "predz",
            sep=",",
            file=self.overall_right_pred,
            flush=True,
        )

        print(
            "eval_exp",
            "cur_exp",
            "action",
            "absdx",
            "absdyaw",
            "absdz",
            sep=",",
            file=self.perexp_forward_absdelta,
            flush=True,
        )
        print(
            "eval_exp",
            "cur_exp",
            "action",
            "absdx",
            "absdyaw",
            "absdz",
            sep=",",
            file=self.perexp_left_absdelta,
            flush=True,
        )
        print(
            "eval_exp",
            "cur_exp",
            "action",
            "absdx",
            "absdyaw",
            "absdz",
            sep=",",
            file=self.perexp_right_absdelta,
            flush=True,
        )
        print(
            "eval_exp",
            "cur_exp",
            "action",
            "absdx",
            "absdyaw",
            "absdz",
            sep=",",
            file=self.perexp_uni_absdelta,
            flush=True,
        )

    def overall_pred_metrics(
        self, eval_exp, action, predx, predyaw, predz
    ):
        if action == 1:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.overall_forward_pred,
                flush=True,
            )
        elif action == 2:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.overall_left_pred,
                flush=True,
            )
        elif action == 3:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.overall_right_pred,
                flush=True,
            )


    def perexp_absdelta(
        self, eval_exp, action, predx, predyaw, predz
    ):
        if action == 1:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.perexp_forward_absdelta,
                flush=True,
            )
        elif action == 2:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.perexp_left_absdelta,
                flush=True,
            )
        elif action == 3:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.perexp_right_absdelta,
                flush=True,
            )
        elif action == 0:
            print(
                eval_exp,
                self.cur_exp,
                action,
                predx,
                predyaw,
                predz,
                sep=",",
                file=self.perexp_uni_absdelta,
                flush=True,
            )

    def wandblogger(self,loss_result, eval_flag, scence_num,epochs):
        if eval_flag == 1:
            wandb.log({f"test/experience{scence_num}_loss": loss_result[0]})
            wandb.log({f"test_abs/experience{scence_num}_abs_dx": loss_result[1]})
            wandb.log({f"test_abs/experience{scence_num}_abs_dz": loss_result[2]})
            wandb.log({f"test_abs/experience{scence_num}_abs_dyaw": loss_result[3]})

            wandb.log({f"test_forward/experience{scence_num}_forward_dx": loss_result[4]})
            wandb.log({f"test_forward/experience{scence_num}_forward_dz": loss_result[5]})
            wandb.log({f"test_forward/experience{scence_num}_forward_dyaw": loss_result[6]})

            wandb.log({f"test_left/experience{scence_num}_left_dx": loss_result[7]})
            wandb.log({f"test_left/experience{scence_num}_left_dz": loss_result[8]})
            wandb.log({f"test_left/experience{scence_num}_left_dyaw": loss_result[9]})

            wandb.log({f"test_right/experience{scence_num}_right_dx": loss_result[10]})
            wandb.log({f"test_right/experience{scence_num}_right_dz": loss_result[11]})
            wandb.log({f"test_right/experience{scence_num}_right_dyaw": loss_result[12]})

            self.perexp_absdelta(scence_num, 0, loss_result[1], loss_result[2], loss_result[3])

            self.perexp_absdelta(scence_num, 1, loss_result[4], loss_result[5], loss_result[6])

            self.perexp_absdelta(scence_num, 2, loss_result[7], loss_result[8], loss_result[9])

            self.perexp_absdelta(scence_num, 3, loss_result[10], loss_result[11], loss_result[12])

        elif eval_flag == 0:
            wandb.log({f"train_loss/all_general_loss": loss_result[0]})
            #wandb.log({f"train/experience{scence_num}_general_loss": loss_result[0]})
            #wandb.log({f"train/experience{scence_num}_abs_dx": loss_result[1]})
            #wandb.log({f"train/experience{scence_num}_abs_dz": loss_result[2]})
            #wandb.log({f"train/experience{scence_num}_abs_dyaw": loss_result[3]})
        elif eval_flag == -1:
            wandb.log({f"validation_loss/experience{scence_num}_loss_by_epochs": loss_result[0]})
            wandb.log({f"validation/experience{scence_num}_abs_dx": loss_result[1]})
            wandb.log({f"validation/experience{scence_num}_abs_dz": loss_result[2]})
            wandb.log({f"validation/experience{scence_num}_abs_dyaw": loss_result[3]})

    def compute_abs_diff(
            self,
            pred_deltax,
            pred_deltaz,
            pred_deltayaw,
            action,
            gt_deltax,
            gt_deltaz,
            gt_deltayaw,
            eval_exp,
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

        forward_pred_deltax = []
        forward_pred_deltaz = []
        forward_pred_deltayaw = []
        left_pred_deltax = []
        left_pred_deltaz = []
        left_pred_deltayaw = []
        right_pred_deltax = []
        right_pred_deltaz = []
        right_pred_deltayaw = []

        for i in range(len(action)):
            if action[i] == 1:
                forward_pred_deltax.append(delta_x_diffs[i])
                forward_pred_deltaz.append(delta_z_diffs[i])
                forward_pred_deltayaw.append(delta_yaw_diffs[i])

                self.overall_pred_metrics(eval_exp, action[i].item(),
                                          pred_deltax[i].item(),pred_deltayaw[i].item(), pred_deltaz[i].item())

            elif action[i] == 2:
                left_pred_deltax.append(delta_x_diffs[i])
                left_pred_deltaz.append(delta_z_diffs[i])
                left_pred_deltayaw.append(delta_yaw_diffs[i])
                self.overall_pred_metrics(eval_exp, action[i].item(),
                                          pred_deltax[i].item(),pred_deltayaw[i].item(), pred_deltaz[i].item())
            elif action[i] == 3:
                right_pred_deltax.append(delta_x_diffs[i])
                right_pred_deltaz.append(delta_z_diffs[i])
                right_pred_deltayaw.append(delta_yaw_diffs[i])

                self.overall_pred_metrics(eval_exp, action[i].item(),
                                          pred_deltax[i].item(),pred_deltayaw[i].item(), pred_deltaz[i].item())

        forward_mean_dx = torch.mean(torch.tensor(forward_pred_deltax))
        forward_mean_dz = torch.mean(torch.tensor(forward_pred_deltaz))
        forward_mean_dyaw = torch.mean(torch.tensor(forward_pred_deltayaw))

        left_mean_dx = torch.mean(torch.tensor(left_pred_deltax))
        left_mean_dz = torch.mean(torch.tensor(left_pred_deltaz))
        left_mean_dyaw = torch.mean(torch.tensor(left_pred_deltayaw))

        right_mean_dx = torch.mean(torch.tensor(right_pred_deltax))
        right_mean_dz = torch.mean(torch.tensor(right_pred_deltaz))
        right_mean_dyaw = torch.mean(torch.tensor(right_pred_deltayaw))

        return torch.mean(delta_x_diffs), torch.mean(delta_z_diffs), torch.mean(
            delta_yaw_diffs), forward_mean_dx, forward_mean_dz, forward_mean_dyaw, left_mean_dx, left_mean_dz, left_mean_dyaw, right_mean_dx, right_mean_dz, right_mean_dyaw


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
                                strategy.mb_y[:, 0].unsqueeze(1),
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
        self.wandblogger(result_loss_epoch, 0, strategy.experience.current_experience, self.current_epoch_num)

        self.current_epoch_num = self.current_epoch_num + 1
        del result_loss_epoch
        del result_loss
        gc.collect()
        torch.cuda.empty_cache()
        pass

    def after_training_exp(self, strategy: 'PluggableStrategy'):
        os.makedirs(RESUME_PATH, exist_ok=True)
        # 拼接保存路径
        #save_path = os.path.join(RESUME_PATH, "naive_Exp{}_resumetwotime.pth".format(str(strategy.experience.current_experience)))

        save_path = os.path.join(RESUME_PATH,
                                 "{}Exp{}_resume{}time.pth".format(TRAIN, str(strategy.experience.current_experience),
                                                                   TIMES))
        # 保存模型
        torch.save(strategy.model.state_dict(), save_path)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        del self.currentexp_train_epoch_iter_info
        gc.collect()
        self.currentexp_train_epoch_iter_info = {}
        self.current_epoch_num = 0


    def after_eval_iteration(self, strategy: 'PluggableStrategy'):
        abs_dx,abs_dz,abs_dyaw,fpx,fpz,fpyaw,lpx,lpz,lpyaw,rpx,rpz,rpyaw = self.compute_abs_diff(strategy.mb_output[:, 0].unsqueeze(1),
                                strategy.mb_output[:, 1].unsqueeze(1),
                                strategy.mb_output[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 0].unsqueeze(1),
                                strategy.mb_y[:, 1].unsqueeze(1),
                                strategy.mb_y[:, 2].unsqueeze(1),
                                strategy.mb_y[:, 3].unsqueeze(1),
                                strategy.experience.current_experience)
        self.currentexp_eval_iter_info.append([strategy.loss.item(), abs_dx.item(), abs_dz.item(), abs_dyaw.item(),fpx.item(),fpz.item(),fpyaw.item(),lpx.item(),lpz.item(),lpyaw.item(),rpx.item(),rpz.item(),rpyaw.item()])

    def after_eval_exp(self, strategy: 'PluggableS'
                                       'trategy'):
        result_loss = np.array(self.currentexp_eval_iter_info)
        result_loss = np.mean(result_loss, axis=0)
        #print("当前准备记录validation,epoch是 ",self.current_epoch_num)
        datasize = len(strategy.adapted_dataset)
        if datasize == 1024:
             self.wandblogger(result_loss, -1, strategy.experience.current_experience,self.current_epoch_num)
        elif datasize == 2048:
             self.wandblogger(result_loss, 1, strategy.experience.current_experience,self.current_epoch_num)
        else:
            print("some thing wrong happen")


        del self.currentexp_eval_iter_info
        gc.collect()
        torch.cuda.empty_cache()
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
        normalize_visual_inputs=NORMALIZE,
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

        if self._n_input_depth > 0:
            depth_observations = observation_pairs[:, :, :, -2:]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(
                [
                    depth_observations[:, : self._n_input_depth // 2, :],
                    depth_observations[:, self._n_input_depth // 2 :, :],
                ]
            )
        # input order:
        #[0 is pre_rgb; 1 is prev_depth ; 2 is cur_rgb ; 3 cur_depth]
        # [prev_rgb, prev_depth, prev_discretized_depth, prev_top_down_view,
        #  cur_rgb, cur_depth, cur_discretized_depth, cur_top_down_view]
        cnn_input = [j for i in list(zip(*cnn_input)) for j in i]

        x = torch.cat(cnn_input, dim=1)
        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        del cnn_input
        return x

class VisualOdometryCNNBase(nn.Module):
    def __init__(
        self,
        *,
        observation_space=OBSERVATION_SPACE,
        observation_size=OBSERVATION_SIZE,
        hidden_size=512,
        resnet_baseplanes=64,
        backbone="resnet50",
        normalize_visual_inputs=NORMALIZE,
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
    seed_value = 10777140
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(DEVICE)


    #model = VisualOdometryCNNBase()
    model = vision_transformer.vit_small_patch16_224(
        pretrained=False,
        img_size=(320, 160),
        in_chans=3,
        splite_head=False,
        class_token=True,
        normalize_visual_inputs=False,
        global_pool='avg',
        num_classes=3
    )
    file_names = os.listdir(RESUME_PATH)
    # 筛选出以".pth"结尾的文件并按EXP数字排序
    pth_files = [file for file in file_names if file.endswith('.pth')]
    pth_files.sort(key=lambda x: int(x.split('_')[4][6:]))
    print(pth_files)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    initexperience = 0
    if ESUME_TRAINR:
        for pth_file in pth_files[initexperience:]:

            file_path = os.path.join(RESUME_PATH, pth_file)
            if ESUME_TRAINR:
                # 使用正则表达式提取信息
                match = re.match(r".*Exp(\d+)_", pth_file)
                if match:
                    exp_number = int(match.group(1))
                    initexperience = exp_number + 1
                    print("now we go from : ")
                    print(initexperience)
                    print(initexperience)
                else:
                    print("未找到匹配的模式")
                    initexperience = -1
                model.load_state_dict(torch.load(file_path))

            optimizer = optimizer_continue(model.to(device), "Adam")
            criterion = predict_diff_loss()

            pjn = "VIT-TEST-72EXP"
            wb_logger = WandBLogger(
                project_name=pjn,  # set the wandb project where this run will be logged
                # track hyperparameters and run metadata
                config={
                    "learning_rate": VOTRAIN_LR,
                    "Nor": NORMALIZE,
                    "action_emb": False,
                    "architecture": "CNN-Resnet18-TwoHiddenlayer512",
                    "dataset": "Habitat-Gibson-V2",
                    "epochs": 40,
                })

            # print to stdout
            interactive_logger = InteractiveLogger()
            # 根据参数构建一个标识符
            params_identifier = f"{TRAIN}_Test_Evaluation"
            # 获取当前日期和时间
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 构建log文件夹路径
            # log_folder = f"{RESUME_PATH}/csvfile/{current_datetime}_{params_identifier}/"
            log_folder = f"{RESUME_PATH}/csvfile/"

            # 现在你可以将log_folder传递给CSVLogger
            csv_logger = CSVLogger(log_folder=log_folder)
            text_logger = TextLogger(open(f"{log_folder}log.txt", "a"))
            log_folder = f"{RESUME_PATH}/csvfile/"
            custom_plugin = CustomSavePlugin(log_folder=log_folder, cur_exp=initexperience - 1)
            eval_plugin = EvaluationPlugin(
                # EWCPlugin(ewc_lambda=0.25),
                # LwFPlugin(alpha=0.5, temperature=0.1),
                custom_plugin,
                # early.EarlyStoppingPlugin(patience=5,val_stream_name="test_stream",metric_name="Loss_Stream",mode="min",peval_mode="epoch",margin=0.0,verbose=True),
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
                # ,wb_logger
                loggers=[interactive_logger, text_logger, csv_logger],
                collect_all=True,
            )

            cl_strategy = Naive(
                model=model,
                train_mb_size=32,
                optimizer=optimizer,
                criterion=criterion,
                evaluator=eval_plugin,
                train_epochs=40,
                device=device,
                eval_every=1
            )
            benchmark, test_benchmark = avl_data_set(False,DATA_FOLDER_PATH,device)
            print("Training & validation completed,test starting")
            print("现在将要测试的是")
            print(initexperience)
            print(initexperience)
            cl_strategy.eval(test_benchmark.test_stream, shuffle=False)
            print("Evaluation completed")




if __name__ == "__main__":
    main()

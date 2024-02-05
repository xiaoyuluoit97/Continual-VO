import gc
import os
import random
import numpy as np

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
from avalanche.training.supervised import Naive,EWC,LwF
from avalanche.training.templates import SupervisedTemplate
#from kubernetesdlprofile import kubeprofiler
from avalanche.evaluation.metrics import (
    timing_metrics,
    loss_metrics
)


class JointCustomSavePlugin(PluginMetric):
    def __init__(self,times,train,resume_path):
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
        self.times=times
        self.train_name=train
        self.resume_path=resume_path

    def wandblogger(self, loss_result, eval_flag, scence_num, epochs):
        if eval_flag == 1:
            wandb.log({f"test_general_loss/all_general_loss": loss_result[0]})
            wandb.log({f"test_general_loss/all_general_abs_dx": loss_result[1]})
            wandb.log({f"test_general_loss/all_general_abs_dz": loss_result[2]})
            wandb.log({f"test_general_loss/all_general_abs_dyaw": loss_result[3]})

            #wandb.log({f"test_loss/experience{scence_num}_loss": loss_result[0]})
            #wandb.log({f"test/experience{scence_num}_abs_dx": loss_result[1]})
            #wandb.log({f"test/experience{scence_num}_abs_dz": loss_result[2]})
            #wandb.log({f"test/experience{scence_num}_abs_dyaw": loss_result[3]})
        elif eval_flag == 0:
            wandb.log({f"train_loss/all_general_loss": loss_result[0]})
            wandb.log({f"train_general_loss/all_general_loss": loss_result[0]})
            wandb.log({f"train_general_loss/all_general_abs_dx": loss_result[1]})
            wandb.log({f"train_general_loss/all_general_abs_dz": loss_result[2]})
            wandb.log({f"train_general_loss/all_general_abs_dyaw": loss_result[3]})



            #wandb.log({f"train/experience{scence_num}_general_loss": loss_result[0]})
            #wandb.log({f"train/experience{scence_num}_abs_dx": loss_result[1]})
            #wandb.log({f"train/experience{scence_num}_abs_dz": loss_result[2]})
            #wandb.log({f"train/experience{scence_num}_abs_dyaw": loss_result[3]})
        elif eval_flag == -1:
            wandb.log({f"validation_loss/all_general_loss": loss_result[0]})
            wandb.log({f"validation_loss/all_general_loss": loss_result[0]})
            wandb.log({f"validation_loss/all_general_abs_dx": loss_result[1]})
            wandb.log({f"validation_loss/all_general_abs_dz": loss_result[2]})
            wandb.log({f"validation_loss/all_general_abs_dyaw": loss_result[3]})

            #wandb.log({f"validation_loss/experience{scence_num}_loss_by_epochs": loss_result[0]})
            #wandb.log({f"validation/experience{scence_num}_abs_dx": loss_result[1]})
            #wandb.log({f"validation/experience{scence_num}_abs_dz": loss_result[2]})
            #wandb.log({f"validation/experience{scence_num}_abs_dyaw": loss_result[3]})

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

        return torch.mean(delta_x_diffs), torch.mean(delta_z_diffs), torch.mean(delta_yaw_diffs)

    def result(self, **kwargs):
        pass

    def reset(self, **kwargs) -> None:
        pass

    def before_training_epoch(
            self, strategy: "SupervisedTemplate"
    ):
        self.currentexp_train_epoch_iter_info[self.current_epoch_num] = []

    def after_training_iteration(
            self, strategy: "SupervisedTemplate"
    ):
        abs_dx, abs_dz, abs_dyaw = self.compute_abs_diff(strategy.mb_output[:, 0].unsqueeze(1),
                                                         strategy.mb_output[:, 1].unsqueeze(1),
                                                         strategy.mb_output[:, 2].unsqueeze(1),
                                                         strategy.mb_y[:, 1].unsqueeze(1),
                                                         strategy.mb_y[:, 2].unsqueeze(1),
                                                         strategy.mb_y[:, 3].unsqueeze(1))
        # print("now is scencec"+str(strategy.mb_y[:,6].unsqueeze(1)[0]))
        self.currentexp_train_epoch_iter_info[self.current_epoch_num].append(
            [strategy.loss.item(), abs_dx.item(), abs_dz.item(), abs_dyaw.item()])

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        os.makedirs(self.resume_path, exist_ok=True)
        # 拼接保存路径
        # save_path = os.path.join(RESUME_PATH, "naive_Exp{}_resumetwotime.pth".format(str(strategy.experience.current_experience)))

        save_path = os.path.join(self.resume_path,
                                 "{}epoch{}_resume{}time.pth".format(self.train_name, str(self.current_epoch_num),
                                                                   self.times))
        torch.save(strategy.model.state_dict(), save_path)

        result_loss = np.array(self.currentexp_train_epoch_iter_info[self.current_epoch_num])
        result_loss_epoch = np.mean(result_loss, axis=0)
        self.wandblogger(result_loss_epoch, 0, strategy.experience.current_experience, self.current_epoch_num)

        self.current_epoch_num = self.current_epoch_num + 1

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        del result_loss_epoch
        del result_loss
        gc.collect()
        torch.cuda.empty_cache()
        pass

    def after_training_exp(self, strategy: 'PluggableStrategy'):
        os.makedirs(self.resume_path, exist_ok=True)
        # 拼接保存路径
        # save_path = os.path.join(RESUME_PATH, "naive_Exp{}_resumetwotime.pth".format(str(strategy.experience.current_experience))) str(strategy.experience.current_experience),
                                                                   #self.times))
        # 保存模型
        torch.save(strategy.model.state_dict(), save_path)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        del self.currentexp_train_epoch_iter_info
        gc.collect()
        self.currentexp_train_epoch_iter_info = {}
        self.current_epoch_num = 0

    def after_eval_iteration(self, strategy: 'PluggableStrategy'):
        abs_dx, abs_dz, abs_dyaw = self.compute_abs_diff(strategy.mb_output[:, 0].unsqueeze(1),
                                                         strategy.mb_output[:, 1].unsqueeze(1),
                                                         strategy.mb_output[:, 2].unsqueeze(1),
                                                         strategy.mb_y[:, 1].unsqueeze(1),
                                                         strategy.mb_y[:, 2].unsqueeze(1),
                                                         strategy.mb_y[:, 3].unsqueeze(1))
        self.currentexp_eval_iter_info.append([strategy.loss.item(), abs_dx.item(), abs_dz.item(), abs_dyaw.item()])

    def after_eval_exp(self, strategy: 'PluggableS'
                                       'trategy'):
        result_loss = np.array(self.currentexp_eval_iter_info)
        result_loss = np.mean(result_loss, axis=0)
        # print("当前准备记录validation,epoch是 ",self.current_epoch_num)
        datasize = len(strategy.adapted_dataset)
        if datasize == 1024:
            self.wandblogger(result_loss, -1, strategy.experience.current_experience, self.current_epoch_num)
        elif datasize == 2048:
            self.wandblogger(result_loss, 1, strategy.experience.current_experience, self.current_epoch_num)
        else:
            print("some thing wrong happen")

        del self.currentexp_eval_iter_info
        gc.collect()
        torch.cuda.empty_cache()
        self.currentexp_eval_iter_info = []
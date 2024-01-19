import os
import contextlib
import joblib
import numpy as np
import wandb

import sys
sys.path.append(r"/home/xiaoyu/application/pointnav/PointNav-VO")
import torch
import torch.nn as nn
from torch import Tensor
from vo.common.common_vars import *

DEFAULT_LOSS_WEIGHTS = {"dx": 1.0, "dz": 1.0, "dyaw": 1.0}
# [x, z, w]
DEFAULT_DELTA_TYPES = ["dx", "dz", "dyaw"]
EPSILON = 1e-8

LOSS_WEIGHTS_FIXED = 0

UNIFIED = -1
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

NO_NOISE_DELTAS = {
    MOVE_FORWARD: [0.0, -0.25, 0.0],
    TURN_LEFT: [0.0, 0.0, np.radians(30)],
    TURN_RIGHT: [0.0, 0.0, -np.radians(30)],
}

def wandblogger(actions,abs_dx,abs_dz,abs_dyaw,eval_flag,scence_num,loss):
    if eval_flag==1:
        wandb.log({"eval/scence{scence_num}_loss": loss})
        if len(actions) == 1:
            if actions == 1:
                wandb.log({f"eval/scence{scence_num}_forward_dx": abs_dx, f"eval/scence{scence_num}_forward_dz": abs_dz, f"eval/scence{scence_num}_forward_dyaw": abs_dyaw})
            elif actions == 2:
                wandb.log({f"eval/scence{scence_num}_right_dx": abs_dx, f"eval/scence{scence_num}_right_dz": abs_dz, f"eval/scence{scence_num}_right_dyaw": abs_dyaw})
            elif actions == 3:
                wandb.log({f"eval/scence{scence_num}_left_dx": abs_dx, f"eval/scence{scence_num}_left_dz": abs_dz, f"eval/scence{scence_num}_left_dyaw": abs_dyaw})
        else:
            wandb.log({f"eval/scence{scence_num}_abs_dx": abs_dx})
            wandb.log({f"eval/scence{scence_num}_abs_dz": abs_dz})
            wandb.log({f"eval/scence{scence_num}_abs_dyaw": abs_dyaw})
    elif eval_flag==0:
        wandb.log({"train/scence{scence_num}_general_loss": loss})
        if len(actions) == 1:
            if actions == 1:
                wandb.log({f"train/scence{scence_num}_forward_dx": abs_dx, f"train/scence{scence_num}_forward_dz": abs_dz, f"train/scence{scence_num}_forward_dyaw": abs_dyaw})
            elif actions == 2:
                wandb.log({f"train/scence{scence_num}_right_dx": abs_dx, f"train/scence{scence_num}_right_dz": abs_dz, f"train/scence{scence_num}_right_dyaw": abs_dyaw})
            elif actions == 3:
                wandb.log({f"train/scence{scence_num}_left_dx": abs_dx, f"train/scence{scence_num}_left_dz": abs_dz, f"train/scence{scence_num}_left_dyaw": abs_dyaw})
        else:
            wandb.log({f"train/scence{scence_num}_abs_dx": abs_dx})
            wandb.log({f"train/scence{scence_num}_abs_dz": abs_dz})
            wandb.log({f"train/scence{scence_num}_abs_dyaw": abs_dyaw})


class predict_diff_loss(nn.modules.loss._Loss):
    def __init__(self, weight=None, size_average=True):
        super(predict_diff_loss, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        pred_deltax = input[:, 0].unsqueeze(1)
        pred_deltaz = input[:, 1].unsqueeze(1)
        pred_deltayaw = input[:, 2].unsqueeze(1)

        actions = target[:, 0].unsqueeze(1)

        gt_deltax = target[:, 1].unsqueeze(1)
        gt_deltaz = target[:, 2].unsqueeze(1)
        gt_deltayaw = target[:, 3].unsqueeze(1)
        dz_regress_masks = target[:, 4].unsqueeze(1)
        eval_flag = target[:, 5].unsqueeze(1)
        scence_num = target[:, 6].unsqueeze(1)
        loss_weights = compute_loss_weights(
            actions, gt_deltax, gt_deltaz, gt_deltayaw
        )
        loss = _compute_loss(pred_deltax,pred_deltaz,pred_deltayaw,gt_deltax,gt_deltaz,gt_deltayaw,loss_weights,dz_regress_masks)
        #wandblogger(actions, abs_dx*100, abs_dz*100, abs_dyaw*100,eval_flag,scence_num.item(),loss)
        return loss

def _process_batch(batch_data):
    (
        data_types,
        raw_rgb_pairs,
        raw_depth_pairs,
        raw_discretized_depth_pairs,
        raw_top_down_view_pairs,
        actions,
        delta_xs,
        delta_ys,
        delta_zs,
        delta_yaws,
        dz_regress_masks,
        chunk_idxs,
        entry_idxs,
    ) = batch_data

   # print(batch_data)
    return actions

def lossfunction_pertoper():
    return 0

def _compute_loss(
        pred_deltax,
        pred_deltaz,
        pred_deltayaw,
        gt_deltax,
        gt_deltaz,
        gt_deltayaw,
        loss_weights,
        dz_regress_masks=None,
):
    ##here is the target

    # NOTE: we should not use sqrt in the loss
    # since it may cause NaN in the backward
    assert pred_deltax.size() == gt_deltax.size()
    delta_x_diffs = (gt_deltax - pred_deltax) ** 2
    loss_dx = torch.mean(delta_x_diffs * loss_weights["dx"])



    assert pred_deltaz.size() == gt_deltaz.size()
    delta_z_diffs = (gt_deltaz - pred_deltaz) ** 2

    if dz_regress_masks is not None:
        assert (
                gt_deltaz.size() == dz_regress_masks.size()
        )
        delta_z_diffs = dz_regress_masks * delta_z_diffs
        filtered_dz_idxes = torch.nonzero(
            dz_regress_masks == 1.0, as_tuple=True
        )[0]
    else:
        filtered_dz_idxes = torch.tensor(np.arange(gt_deltaz.size()[0]))
    loss_dz = torch.mean(delta_z_diffs * loss_weights["dz"])

    #return loss_dz, abs_diff_dz, target_magnitude_dz, relative_diff_dz
    assert pred_deltayaw.size() == gt_deltayaw.size()
    delta_yaw_diffs = (gt_deltayaw - pred_deltayaw) ** 2
    loss_dyaw = torch.mean(delta_yaw_diffs * loss_weights["dyaw"])




    loss_all = loss_dx + loss_dz + loss_dyaw
    return loss_all


def compute_loss_weights(actions, dxs, dzs, dyaws):
        if (
                LOSS_WEIGHTS_FIXED == 1
        ):
            loss_weights = {
                k: torch.ones(dxs.size()).to(dxs.device) * v
                for k, v in DEFAULT_LOSS_WEIGHTS.items()
            }
        else:
            no_noise_ds = np.array([NO_NOISE_DELTAS[int(_)] for _ in actions])
            no_noise_ds = torch.from_numpy(no_noise_ds).float().to(dxs.device)

            loss_weights = {}
            multiplier = DEFAULT_LOSS_WEIGHTS
            loss_weights["dx"] = torch.exp(
                multiplier["dx"] * torch.abs(no_noise_ds[:, 0].unsqueeze(1) - dxs)
            )
            loss_weights["dz"] = torch.exp(
                multiplier["dz"] * torch.abs(no_noise_ds[:, 1].unsqueeze(1) - dzs)
            )
            loss_weights["dyaw"] = torch.exp(
                multiplier["dyaw"] * torch.abs(no_noise_ds[:, 2].unsqueeze(1) - dyaws)
            )

            for v in loss_weights.values():
                torch.all(v >= 1.0)

        return loss_weights
from torch.utils.data import Dataset, IterableDataset,SequentialSampler
from avalanche.benchmarks.utils.dataset_definitions import IDatasetWithTargets,IDataset,ITensorDataset
import sys
from torch import FloatTensor
from avalanche.benchmarks.scenarios.generic_benchmark_creation import LazyStreamDefinition,create_lazy_generic_benchmark
import gc
import h5py
import numpy as np
from tqdm import tqdm
import torch
from habitat import Config, logger
import random
from avalanche.benchmarks.utils import (
    make_classification_dataset,
)
from avalanche.benchmarks.scenarios.generic_scenario import (
    CLExperience,
    CLStream,
    CLScenario,
    DatasetExperience,
    CLScenario,
)
from avalanche.benchmarks.scenarios.online_scenario import OnlineCLScenario
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Generic,
    overload,
)

import os
import threading
import time
import copy
import re
TCLDataset = TypeVar("TCLDataset", bound="AvalancheDataset")
TClassificationDataset = TypeVar(
    "TClassificationDataset", bound="ClassificationDataset"
)
TCLScenario = TypeVar("TCLScenario", bound="CLScenario")
TDatasetScenario = TypeVar("TDatasetScenario", bound="DatasetScenario")
TOnlineCLScenario = TypeVar("TOnlineCLScenario", bound="OnlineCLScenario")
TCLStream = TypeVar("TCLStream", bound="CLStream")
TCLExperience = TypeVar("TCLExperience", bound="CLExperience")
TOnlineCLExperience = TypeVar("TOnlineCLExperience", bound="OnlineCLExperience")
TOnlineClassificationExperience = TypeVar(
    "TOnlineClassificationExperience", bound="OnlineClassificationExperience"
)

CUR_REL_TO_PREV = 0
TRAIN_FILE_LIST = []
TEST_FILE_LIST = []
EVAL_FILE_LIST = []
VIS_SIZE_W = 341
VIS_SIZE_H = 192
CHUNK_NUM_LOAD_MORE = 2

FORWARD_ACT_CHANNLE = torch.full((192, 341, 1), 1)
LEFT_ACT_CHANNLE = torch.full((192, 341, 1), 0)
RIGHT_ACT_CHANNLE = torch.full((192, 341, 1), 2)


def get_num_scence(string):
    match = re.search(r"\d+", string)  # 正则表达式匹配数字部分
    if match:
        number = int(match.group())  # 提取匹配到的数字
        return number
def get_num(string):
    # 提取字符串中的num部分作为排序依据
    return int(string.split("_")[-1])

# IDatasetWithTargets
class CLPairDataset():
    def __init__(
            self,
            _data_f,
            eval_flag,
            scence_num,
            chunk_size=None,
            _chunk_splits=None,
            index_exist=None,
            loaded_indices=None,
            rgb_d_pairs=None,
            data_in_memory=None,
            validate_index_inmemory=None,
            target=None,

            num_workers=0,
            act_type=-1,
            collision="-1",
            discretize_depth="none",
            discretized_depth_channels=0,
            gen_top_down_view=False,
            top_down_view_infos={},
            geo_invariance_types=[],
            partial_data_n_splits=1,
            # data_aug=False,
    ):


        self.data = {}  # 用于存储已读取的数据
        self.rgb_d_pairs = {}
        self.targets = {}
        self.loaded_indices = set()  # 已读取的索引集合
        self._data_f = _data_f
        self._eval = eval_flag
        self._scence_num = scence_num
        self._current_training_chunk = 0
        self._current_loading_chunk = 0
        self.chunk_in_memory = set()
        self.device = torch.device("cuda")
        #0 means do not do any load action(not start yet)
        #1 means loading (in process)
        #2 means loaded (in memory)
        self.chunk_state = {}
        self._act_type = act_type
        self._collision = collision
        self._geo_invariance_types = geo_invariance_types
        self._len = 0
        self._act_left_right_len = 0

        #thread thing
        self.training_event = threading.Event()
        self.loading_event = threading.Event()


        self._partial_data_n_splits = partial_data_n_splits

        self._gen_top_down_view = gen_top_down_view
        self._top_down_view_infos = top_down_view_infos

        self.memory_loading_lock = threading.Lock()
        # # RGB stored with uint8
        # self._rgb_pair_size = 2 * self._vis_size_w * self._vis_size_h * 3
        # # Depth stored with float16
        # self._depth_pair_size = 2 * self._vis_size_w * self._vis_size_h * 2

        with h5py.File(self._data_f, "r", libver="latest") as f:
            self._chunk_size = f[list(f.keys())[0]]["prev_rgbs"].shape[0]

        # # for rgb + depth
        # self._chunk_bytes = int(
        #     np.ceil((self._rgb_pair_size + self._depth_pair_size) * self._chunk_size)
        # )
        # # for misc information
        # self._chunk_bytes += 20 * 2
        # logger.info(f"\nDataset: chunk bytes {self._chunk_bytes / (1024 * 1024)} MB\n")


        logger.info("Get index mapping from h5py ...")
        self.length = 0
        self._chunk_splits = []

        with h5py.File(self._data_f, "r", libver="latest") as f:
            for scence_n_chunk_k in tqdm(sorted(f.keys())):
                self._chunk_splits.append(scence_n_chunk_k)
                # print(scence_n_chunk_k)
                valid_idxes = self._indexes(f, scence_n_chunk_k)
                self.length += len(valid_idxes)
        # print(length)

        random.shuffle(self._chunk_splits)
        print(self._chunk_splits)
        new_list = []
        new_order = 0
        for chunk in self._chunk_splits:
            chunk = "{}_{}".format(chunk.split("_")[0], new_order)
            new_order = new_order + 1
            new_list.append(chunk)
        self._chunk_splits = new_list
        print(self._chunk_splits)

        logger.info("... done.\n")



    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self._memory_controll(index)

    def _indexes(self, h5_f, scence_n_chunk_k):
        valid_act_idxes = np.arange(h5_f[scence_n_chunk_k]["actions"].shape[0])
        return list(valid_act_idxes)

    def wait_loading(self, num_of_chunk_training):
        # waiting for loading operation
        while num_of_chunk_training not in self.chunk_state or self.chunk_state[num_of_chunk_training] != 2:
            logger.info("读得太慢")
            self.training_event.wait()
            #self.loading_event.set()

    def wait_training(self, num_of_chunk_loading):
        # waiting for training operation
        while num_of_chunk_loading > self._current_training_chunk + CHUNK_NUM_LOAD_MORE:
            logger.info("训练太慢")
            self.loading_event.wait()
            self.training_event.set()

    def training_condition_decide(self):
        # waiting for loading operation
        if self.chunk_state[self._current_training_chunk] == 2:
            print("读得太慢")
            self.training_event.set()

    def loading_condition_decide(self):
        if self._current_loading_chunk > self._current_training_chunk + CHUNK_NUM_LOAD_MORE:
            print("目前读取的chunk大于训练的chunk和读取总和了，该停止了",self._current_loading_chunk,self._current_training_chunk)

            self.loading_event.clear()
            self.loading_event.wait()

        else:

            print("读得太慢")
            self.loading_event.set()

    def _memory_controll(self, index):

        if index == 0:
            #print("当前index为0")
            self._current_training_chunk = 0
            self._current_loading_chunk = 0
            my_thread = threading.Thread(target=self._load_chunk_into_memory)
            my_thread.start()
            self.training_event.set()
            self.loading_event.set()


        #print("the index from avalanche want loaded right now",index)
        num_of_chunk = int(index / self._chunk_size)
        #ind = index - int(index / self._chunk_size) * self._chunk_size
        ind = index % self._chunk_size
        self._current_training_chunk = num_of_chunk
        if ind == 0:
            #print("新的chunk到了")
            self.loading_condition_decide()
        if num_of_chunk > 0 and ind == 0:
            self._remove_data_from_memory(num_of_chunk-1)

        if index == self.length-1:
            logger.info("last one data point")
            self.training_event.clear()
            self.loading_event.clear()
            if num_of_chunk not in self.chunk_state or self.chunk_state[num_of_chunk] == 1:
                self.training_event.wait()
                time.sleep(3)
                rgb_d_pairs = copy.deepcopy(self.rgb_d_pairs[num_of_chunk][ind])
                targets = copy.deepcopy(self.targets[num_of_chunk][ind])
                self._remove_data_from_memory(num_of_chunk)
                #print(len(rgb_d_pairs))
                return rgb_d_pairs, targets
            elif self.chunk_state[num_of_chunk] == 2:
                rgb_d_pairs = copy.deepcopy(self.rgb_d_pairs[num_of_chunk][ind])
                targets = copy.deepcopy(self.targets[num_of_chunk][ind])
                self._remove_data_from_memory(num_of_chunk)
                #print(len(rgb_d_pairs))
                return rgb_d_pairs, targets


        #key not exist yet
        if num_of_chunk not in self.chunk_state or self.chunk_state[num_of_chunk] == 1:
            while num_of_chunk not in self.chunk_state or self.chunk_state[num_of_chunk] != 2:
                time.sleep(0.5)
            self.training_event.wait()
            return self.rgb_d_pairs[num_of_chunk][ind], self.targets[num_of_chunk][ind]
        elif self.chunk_state[num_of_chunk] == 2:
            return self.rgb_d_pairs[num_of_chunk][ind], self.targets[num_of_chunk][ind]

    def _load_chunk_into_memory(self):
        #num_of_chunk = int(index / self._chunk_size)
        with self.memory_loading_lock:
            for chunk_k in self._chunk_splits:
                print(chunk_k)
                number_of_chunk = get_num(chunk_k)
                print(number_of_chunk)
                self._current_loading_chunk = number_of_chunk
                self.loading_condition_decide()
                #print("release the lock,now we can startload chunk", chunk_k)
                self.chunk_state[number_of_chunk] = 1
                with h5py.File(
                        self._data_f,
                        "r",
                        libver="latest",
                        # rdcc_nbytes=self._chunk_bytes,
                        rdcc_nslots=1e7,
                ) as f:
                    # get valid indexes of each chunk!!!!!!!!!
                    index_of_per_chunk = self._indexes(f, chunk_k)
                    self._actions = f[chunk_k]["actions"][()]
                    #self._whole_h5_indexs = f[chunk_k]["whole_h5_indexs"][()]
                    self._prev_rgbs = f[chunk_k]["prev_rgbs"][()]
                    self._cur_rgbs = f[chunk_k]["cur_rgbs"][()]
                    self._prev_depths = f[chunk_k]["prev_depths"][()]
                    self._cur_depths = f[chunk_k]["cur_depths"][()]
                    self._delta_positions = f[chunk_k]["delta_positions"][()]
                    self._delta_rotations = f[chunk_k]["delta_rotations"][()]
                    # for geometric consistency
                    self._prev_global_positions = f[chunk_k]["prev_global_positions"][()]
                    self._prev_global_rotations = f[chunk_k]["prev_global_rotations"][()]
                    self._cur_global_positions = f[chunk_k]["cur_global_positions"][()]
                    self._cur_global_rotations = f[chunk_k]["cur_global_rotations"][()]
                    f.close()
                self.rgb_d_pairs[number_of_chunk] = []
                self.targets[number_of_chunk] = []

                if not self._eval:
                    print("SHUFFLE")
                    random.shuffle(index_of_per_chunk)

                for i, idx in enumerate(index_of_per_chunk):
                    rgb_d_pair, target = self._process_data(number_of_chunk, idx)
                    # 查找某个对象的引用关系
                    self.rgb_d_pairs[number_of_chunk].append(rgb_d_pair)
                    self.targets[number_of_chunk].append(target)


                self.chunk_state[number_of_chunk] = 2
                self.training_condition_decide()
                #print("now chunk", chunk_k, " is loaded")
                del self._actions
                #del self._whole_h5_indexs
                del self._prev_rgbs
                del self._cur_rgbs
                del self._prev_depths
                del self._cur_depths
                del self._delta_positions
                del self._delta_rotations
                # for geometric consistency
                del self._prev_global_positions
                del self._prev_global_rotations
                del self._cur_global_positions
                del self._cur_global_rotations
                gc.collect()
                #print("the rest is clean of chunk", chunk_k)
            #logger.info("thread for loading data is finished")
            pass
        # release the memory.

    def _remove_data_from_memory(self,num_of_chunk):
        del self.rgb_d_pairs[num_of_chunk]
        del self.targets[num_of_chunk]
        del self.chunk_state[num_of_chunk]
        gc.collect()
        #print("\nDelet chunk", num_of_chunk)

    def _process_data(self, scence_n_chunk_k, i):

        data_types = []

        # rgb in HDF5: uint8, reshaped as a vector
        prev_rgb = self._prev_rgbs[i, :].reshape(
            (VIS_SIZE_H, VIS_SIZE_W, 3)
        )
        cur_rgb = self._cur_rgbs[i, :].reshape((VIS_SIZE_H, VIS_SIZE_W, 3))

        # depth in HDF5: float16, reshaped as a vector
        prev_depth = self._prev_depths[i, :].reshape(
            (VIS_SIZE_H, VIS_SIZE_W, 1)
        )
        cur_depth = self._cur_depths[i, :].reshape(
            (VIS_SIZE_H, VIS_SIZE_W, 1)
        )


        #delta states of prev to cur
        delta_pos_cur_rel_to_prev = self._delta_positions[i, :]
        delta_x = FloatTensor([delta_pos_cur_rel_to_prev[0]])
        # delta_yaw = FloatTensor([delta_pos_cur_rel_to_prev[1]])
        delta_z = FloatTensor([delta_pos_cur_rel_to_prev[2]])

        delta_rotation_quaternion_cur_rel_to_prev = self._delta_rotations[i, :]
        # NOTE: must use arctan2 to get correct yaw
        dyaw_cur_rel_to_prev = FloatTensor(
            [
                2
                * np.arctan2(
                    delta_rotation_quaternion_cur_rel_to_prev[1],
                    delta_rotation_quaternion_cur_rel_to_prev[3],
                )
            ]
        )
        delta_yaw = dyaw_cur_rel_to_prev


        #self.loaded_indices.add(self._whole_h5_indexs[i])
        data_types.append(CUR_REL_TO_PREV)
        dz_regress_mask=1

        chunk_idx = int(scence_n_chunk_k)
        entry_idx = i

        eval_flag = torch.tensor([self._eval])
        scence_num = torch.tensor([self._scence_num])
        action = self._actions[i]

        if action == 1:
            acts_channel=FORWARD_ACT_CHANNLE
        elif action == 2:
            acts_channel=LEFT_ACT_CHANNLE
        elif action == 3:
            acts_channel=RIGHT_ACT_CHANNLE

        action = torch.tensor([action]).long()
        dz_regress_mask = torch.tensor([dz_regress_mask]).long()
        chunk_idx = torch.tensor([chunk_idx]).long()
        entry_idx = torch.tensor([entry_idx]).long()

        input_pairs = torch.Tensor(
            np.concatenate([prev_rgb, cur_rgb, prev_depth, cur_depth, acts_channel], axis=2)
        )

        target = torch.cat((action, delta_x, delta_z, delta_yaw, dz_regress_mask, eval_flag,scence_num,chunk_idx, entry_idx))

        return input_pairs,target.squeeze()

def make_dataset_generator(filelist, folder_path,flag):
        #validation
    if flag == "eval":
        num = 1024
        for list in filelist:
            dataset = CLPairDataset(
                _data_f=os.path.join(folder_path, list),
                eval_flag=1,
                scence_num = get_num_scence(list),
                #device = device
            )
            mkd = make_classification_dataset(
                dataset=dataset,
                targets=[0] * num,
                #device=device
            )
            yield mkd
    elif flag =="test":
        num = 1024
        for list in filelist:
            dataset = CLPairDataset(
                _data_f=os.path.join(folder_path, list),
                eval_flag=-1,
                scence_num = get_num_scence(list),
                #device = device
            )
            mkd = make_classification_dataset(
                dataset=dataset,
                targets=[0] * num,
                #device=device
            )
            yield mkd
    else:
        num = 1024
        for list in filelist:
            dataset = CLPairDataset(
                _data_f=os.path.join(folder_path, list),
                eval_flag=0,
                scence_num=get_num_scence(list),
                #device = device
            )
            mkd = make_classification_dataset(
                dataset=dataset,
                targets=[0] * num,
                #device = device
            )
            yield mkd





def joint_train_dataload(ACTION_EMBEDDING,DATA_FOLDER_PATH,device):
    splitenum = 0
    print(device)
    #folder_path = "/tmp/pycharm_project_710/datasetcl"
    folder_path = "/custom/PointNav-VO-single/dataset/vo_dataset"
    files = os.listdir(folder_path)
    # 过滤出以"train"开头的文件名
    train_files = [f for f in files if f.startswith("train")]
    test_files = [f for f in files if f.startswith("test")]
    val_files = [f for f in files if f.startswith("val")]
    # 按结尾的数字进行排序
    def custom_sort(file_name):
        # 提取文件名中第三段的数字
        num = int(file_name.split("_")[2].split(".")[0])
        return num
    # 输出排序后的文件名
    #TRAIN_FILE_LIST= sorted(train_files, key=custom_sort)
    #EVAL_FILE_LIST = sorted(val_files, key=custom_sort)
    #TEST_FILE_LIST = sorted(test_files, key=custom_sort)

    TRAIN_FILE_LIST = sorted(train_files, key=custom_sort)
    EVAL_FILE_LIST = sorted(val_files, key=custom_sort)
    TEST_FILE_LIST = sorted(test_files, key=custom_sort)

    print(TRAIN_FILE_LIST)
    print(EVAL_FILE_LIST)
    print(TEST_FILE_LIST)

    train_dataset = make_dataset_generator(TRAIN_FILE_LIST,folder_path,"train")
    #test_dataset = make_dataset_generator(TEST_FILE_LIST, folder_path, "test")
    eval_dataset = make_dataset_generator(EVAL_FILE_LIST,folder_path,"eval")
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    trainstream = LazyStreamDefinition(
        exps_generator = train_dataset,
        stream_length = 1,
        exps_task_labels = [0],
    )
    #teststream = LazyStreamDefinition(
    #     exps_generator = test_dataset,
    #     stream_length = len(EVAL_FILE_LIST),
    #     exps_task_labels = [0],
    #)
    valstream = LazyStreamDefinition(
        exps_generator = eval_dataset,
        stream_length = 1,
        exps_task_labels = [0],
    )

    benchmark = create_lazy_generic_benchmark(
        train_generator = trainstream,
        test_generator = valstream,
    )
    #test_benchmark = create_lazy_generic_benchmark(
        #train_generator= teststream,
        #test_generator = teststream,
    #)


    return benchmark,None


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())



if __name__ == "__main__":
    main()

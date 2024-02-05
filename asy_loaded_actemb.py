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
TRAIN_BATCH_SIZE = 4
CUR_REL_TO_PREV = 0
TRAIN_FILE_LIST = []
TEST_FILE_LIST = []
EVAL_FILE_LIST = []
VIS_SIZE_W = 341
VIS_SIZE_H = 192
CHUNK_NUM_LOAD_MORE = 3

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
        #self._chunk_fake = ['chunk_0', 'chunk_1', 'chunk_2', 'chunk_3', 'chunk_4', 'chunk_5', 'chunk_6', 'chunk_7']
        with h5py.File(self._data_f, "r", libver="latest") as f:
            #for scence_n_chunk_k in tqdm(self._chunk_fake):
            for scence_n_chunk_k in tqdm(sorted(f.keys())):
                self._chunk_splits.append(scence_n_chunk_k)
                # print(scence_n_chunk_k)
                valid_idxes = self._indexes(f, scence_n_chunk_k)
                self.length += len(valid_idxes)

        # print(length)

        # each chunk in here
        self._chunk_splits = sorted(self._chunk_splits, key=get_num)


        #logger.info("... done.\n")



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
            #logger.info("读得太慢")
            self.training_event.wait()
            #self.loading_event.set()

    def wait_training(self, num_of_chunk_loading):
        # waiting for training operation
        while num_of_chunk_loading > self._current_training_chunk + CHUNK_NUM_LOAD_MORE:
            #logger.info("训练太慢")
            self.loading_event.wait()
            self.training_event.set()

    def training_condition_decide(self):
        # waiting for loading operation
        if self.chunk_state[self._current_training_chunk] == 2:
            #print("读得太慢")
            self.training_event.set()

    def loading_condition_decide(self):
        if self._current_loading_chunk > self._current_training_chunk + CHUNK_NUM_LOAD_MORE:
            print("目前读取的chunk大于训练的chunk和读取总和了，该停止了",self._current_loading_chunk,self._current_training_chunk)
            #print("前Loading event status:", self.loading_event.is_set())
            self.loading_event.clear()
            self.loading_event.wait()
            #print("后Loading event status:", self.loading_event.is_set())
        else:

            #print("读得太慢")
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
        print("要开始动手读取了注意注意注意！")
        with self.memory_loading_lock:
            for chunk_k in self._chunk_splits:
                number_of_chunk = get_num(chunk_k)
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
                    self._whole_h5_indexs = f[chunk_k]["whole_h5_indexs"][()]
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
                    random.shuffle(index_of_per_chunk)
                #print(index_of_per_chunk)
                #time.sleep(12000)
                for i, idx in enumerate(index_of_per_chunk):
                    rgb_d_pair, target = self._process_data(number_of_chunk, idx)
                    # 查找某个对象的引用关系
                    self.rgb_d_pairs[number_of_chunk].append(rgb_d_pair)
                    self.targets[number_of_chunk].append(target)


                self.chunk_state[number_of_chunk] = 2
                self.training_condition_decide()
                #print("now chunk", chunk_k, " is loaded")
                del self._actions
                del self._whole_h5_indexs
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


        self.loaded_indices.add(self._whole_h5_indexs[i])
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
        num = 2048
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
        num = 13888
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





class OnlineCLExperience(DatasetExperience[TCLDataset]):
    """Online CL (OCL) Experience.

    OCL experiences are created by splitting a larger experience. Therefore,
    they keep track of the original experience for logging purposes.
    """

    def __init__(
        self: TOnlineCLExperience,
        current_experience: int,
        origin_stream: CLStream[TOnlineCLExperience],
        benchmark: CLScenario,
        dataset: TCLDataset,
        origin_experience: DatasetExperience,
        subexp_size: int = 1,
        is_first_subexp: bool = False,
        is_last_subexp: bool = False,
        sub_stream_length: Optional[int] = None,
        access_task_boundaries: bool = False,
    ):
        """Init.

        :param current_experience: experience identifier.
        :param origin_stream: origin stream.
        :param origin_experience: origin experience used to create self.
        :param is_first_subexp: whether self is the first in the sub-experiences
            stream.
        :param sub_stream_length: the sub-stream length.
        """
        super().__init__(
            current_experience=current_experience,
            origin_stream=origin_stream,
            benchmark=benchmark,
            dataset=dataset,
        )
        self.access_task_boundaries = access_task_boundaries

        self.origin_experience: DatasetExperience = origin_experience
        self.subexp_size: int = subexp_size
        self.is_first_subexp: bool = is_first_subexp
        self.is_last_subexp: bool = is_last_subexp
        self.sub_stream_length: Optional[int] = sub_stream_length

        self._as_attributes(
            "origin_experience",
            "subexp_size",
            "is_first_subexp",
            "is_last_subexp",
            "sub_stream_length",
            use_in_train=access_task_boundaries,
        )

    @property
    def task_labels(self) -> List[int]:
        return self.origin_experience.task_labels




class OnlineCLExtendScenario(OnlineCLScenario):
    def __init__(
        self,
        #experiences: Iterable[SupervisedClassificationDataset],
        experiences:LazyStreamDefinition,
    ):

        if not isinstance(experiences, Iterable):
            experiences = [experiences]

        online_train_stream = split_online_stream(experiences,4,self)

        streams: List[CLStream] = [online_train_stream]

        # super().__init__(streams=streams)


def fixed_size_experience_split(
    experience: int,
    dataset:CLPairDataset,
    experience_size: int,
    # online_benchmark: TOnlineCLScenario,
    shuffle: bool = True,
    drop_last: bool = False,
    access_task_boundaries: bool = False,
) -> Generator[OnlineCLExperience[TClassificationDataset], None, None]:

    exp_dataset = dataset
    exp_indices = list(range(len(exp_dataset)))
    exp_targets = torch.as_tensor(
        list(exp_dataset.targets), dtype=torch.long  # type: ignore
    )

    if shuffle:
        exp_indices = torch.as_tensor(exp_indices)[
            torch.randperm(len(exp_indices))
        ].tolist()
    sub_stream_length = len(exp_indices) // experience_size
    if not drop_last and len(exp_indices) % experience_size > 0:
        sub_stream_length += 1

    init_idx = 0
    is_first = True
    is_last = False
    exp_idx = 0
    while init_idx < len(exp_indices):
        final_idx = init_idx + experience_size  # Exclusive
        if final_idx > len(exp_indices):
            if drop_last:
                break

            final_idx = len(exp_indices)
            is_last = True

        sub_exp_subset = exp_dataset.subset(exp_indices[init_idx:final_idx])
        sub_exp_targets: torch.Tensor = exp_targets[
            exp_indices[init_idx:final_idx]
        ].unique()

        # origin_stream will be lazily set later
        exp = OnlineCLExperience(
            current_experience=exp_idx,
            origin_stream=None,  # type: ignore
            benchmark=online_benchmark,
            dataset=sub_exp_subset,
            origin_experience=experience,
            classes_in_this_experience=sub_exp_targets.tolist(),
            subexp_size=experience_size,
            is_first_subexp=is_first,
            is_last_subexp=is_last,
            sub_stream_length=sub_stream_length,
            access_task_boundaries=access_task_boundaries,
        )

        is_first = False
        yield exp
        init_idx = final_idx
        exp_idx += 1




def _default_online_split(
    online_benchmark,
    shuffle: bool,
    drop_last: bool,
    access_task_boundaries: bool,
    exp: DatasetExperience[TClassificationDataset],
    size: int,
):

    return fixed_size_experience_split(
        experience=exp,
        experience_size=size,
        online_benchmark=online_benchmark,
        shuffle=False,
        drop_last=drop_last,
        access_task_boundaries=access_task_boundaries,
    )


def split_online_stream(
    #original_stream: Iterable[SupervisedClassificationDataset],
    original_stream: LazyStreamDefinition,
    experience_size: int,
    online_benchmark: "OnlineCLScenario[TClassificationDataset]",
    shuffle: bool = True,
    drop_last: bool = False,
    access_task_boundaries: bool = False,
) -> CLStream[DatasetExperience[TClassificationDataset]]:
    print(original_stream)
    def exps_iter():
        for one_exp in original_stream:
            for sub_exp in fixed_size_experience_split(one_exp, 64):
                yield sub_exp

    stream_name: str = getattr(original_stream, "name", "train")
    return CLStream(
        name=stream_name,
        exps_iter=exps_iter(),
        set_stream_info=True,
        benchmark=online_benchmark,
    )

__all__ = [
    "OnlineCLExperience",
    "fixed_size_experience_split",
    "split_online_stream",
    "OnlineCLScenario",
]


def avl_data_set(ACTION_EMBEDDING,DATA_FOLDER_PATH,device):
    splitenum = 72
    print(device)
    #folder_path = "/tmp/pycharm_project_710/datasetcl"
    folder_path = DATA_FOLDER_PATH
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
    test_dataset = make_dataset_generator(TEST_FILE_LIST, folder_path, "test")
    eval_dataset = make_dataset_generator(EVAL_FILE_LIST, folder_path, "eval")
    #eval_dataset = make_dataset_generator(EVAL_FILE_LIST,folder_path,"eval")
    #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    trainstream = LazyStreamDefinition(
        exps_generator = train_dataset,
        stream_length = len(TRAIN_FILE_LIST),
        exps_task_labels = [0] * splitenum,
    )
    teststream = LazyStreamDefinition(
         exps_generator = test_dataset,
         stream_length = len(EVAL_FILE_LIST),
         exps_task_labels = [0] * splitenum,
    )
    valstream = LazyStreamDefinition(
        exps_generator = eval_dataset,
        stream_length = len(TEST_FILE_LIST),
        exps_task_labels = [0] * splitenum,
    )

    benchmark = create_lazy_generic_benchmark(
        train_generator = trainstream,
        test_generator = valstream,
    )
    test_benchmark = create_lazy_generic_benchmark(
        train_generator= teststream,
        test_generator = teststream,
    )


    return benchmark,test_benchmark


def main():
    print(torch.__version__)
    print(torch.cuda.is_available())



if __name__ == "__main__":
    main()

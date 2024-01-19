################################################################################
# Copyright (c) 2022 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-07-2022                                                             #
# Author(s): Antonio Carta                                                     #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
    Datasets with optimized concat/subset operations.
"""
import bisect
import sys
import numpy as np

from avalanche.benchmarks.utils.dataset_utils import (
    slice_alike_object_to_indices,
)

try:
    from collections import Hashable
except ImportError:
    from collections.abc import Hashable

from typing import (
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from torch.utils.data import ConcatDataset
import itertools
from avalanche.benchmarks.utils.dataset_definitions import IDataset

TFlatData = TypeVar("TFlatData", bound="FlatData")
DataT = TypeVar("DataT")
T_co = TypeVar("T_co", covariant=True)


class LazyIndices:
    """More efficient ops for indices.

    Avoid wasteful allocations, accept generators. Convert to list only
    when needed.

    Do not use for anything outside this file.
    """

    def __init__(self, *lists, known_length=None, offset=0):
        new_lists = []
        for ll in lists:
            if isinstance(ll, LazyIndices) and ll._eager_list is not None:
                # already eagerized, don't waste work
                new_lists.append(ll._eager_list)
            else:
                new_lists.append(ll)
        self._lists = new_lists  # freed after eagerification

        if len(new_lists) == 1 and offset == 0:
            self._eager_list = new_lists[0]
        else:
            self._lazy_sequence = itertools.chain(*new_lists)
            """chain of generators
            this will be consumed over time whenever we need elems.
            """
            self._eager_list = None
            """This is the list where we save indices
            whenever we generate them from the lazy sequence.
            """
            self._offset = offset

        # check depth to avoid RecursionError
        if self._depth() > sys.getrecursionlimit() // 4:
            self._to_eager()

        if known_length is not None:
            self._known_length = known_length
        elif self._eager_list is not None:
            self._known_length = len(self._eager_list)
        else:
            self._known_length = sum(len(ll) for ll in new_lists)

    def _depth(self):
        """Return the depth of the LazyIndices tree.
        Use it only to eagerify early to avoid RecursionErrors.
        """
        if self._eager_list is not None:
            return 0

        lens = [0]
        for ll in self._lists:
            if isinstance(ll, LazyIndices):
                lens.append(ll._depth())
        return max(lens) + 1

    def _to_eager(self):
        if self._eager_list is not None:
            return
        self._eager_list = [el + self._offset for el in self._lazy_sequence]
        self._lists = None  # free memory

    def __getitem__(self, item):
        if self._eager_list is None:
            self._to_eager()
        return self._eager_list[item]

    def __add__(self, other):
        return LazyIndices(self, other)

    def __radd__(self, other):
        return LazyIndices(other, self)

    def __len__(self):
        return self._known_length


class LazyRange(LazyIndices):
    """Avoid 'eagerification' step for ranges."""

    def __init__(self, start, end, offset=0):
        self._start = start
        self._end = end
        self._offset = offset
        self._known_length = end - start
        self._eager_list = self

    def _to_eager(self):
        # LazyRange is eager already
        pass

    def __iter__(self):
        for i in range(self._start, self._end):
            yield self._offset + i

    def __getitem__(self, item):
        assert item >= self._start and item < self._end, "LazyRange: index out of range"
        return self._start + self._offset + item

    def __add__(self, other):
        # TODO: could be optimized to merge contiguous ranges
        return LazyIndices(self, other)

    def __len__(self):
        return self._end - self._start


class FlatData(IDataset[T_co], Sequence[T_co]):
    """FlatData is a dataset optimized for efficient repeated concatenation
    and subset operations.

    The class combines concatentation and subsampling operations in a single
    class.

    Class for internal use only. Users shuold use `AvalancheDataset` for data
    or `DataAttribute` for attributes such as class and task labels.

    *Notes for subclassing*

    Cat/Sub operations are "flattened" if possible, which means that they will
    take the datasets and indices from their arguments and create a new dataset
    with them, avoiding the creation of large trees of dataset (what would
    happen with PyTorch datasets). Flattening is not always possible, for
    example if the data contains additional info (e.g. custom transformation
    groups), so subclasses MUST set `can_flatten` properly in order to avoid
    nasty bugs.
    """

    def __init__(
        self,
        datasets: Sequence[IDataset[T_co]],
        indices: Optional[List[int]] = None,
        can_flatten: bool = True,
    ):
        """Constructor

        :param datasets: list of datasets to concatenate.
        :param indices:  list of indices.
        :param can_flatten: if True, enables flattening.
        """
        self._datasets: List[IDataset[T_co]] = list(datasets)
        self._indices: Optional[List[int]] = indices
        self._can_flatten: bool = can_flatten

        if can_flatten:
            self._datasets = _flatten_dataset_list(self._datasets)
            self._datasets, self._indices = _flatten_datasets_and_reindex(
                self._datasets, self._indices
            )
        self._cumulative_sizes = ConcatDataset.cumsum(self._datasets)

        # NOTE: check disabled to avoid slowing down OCL scenarios
        # # check indices
        # if self._indices is not None and len(self) > 0:
        #     assert min(self._indices) >= 0
        #     assert max(self._indices) < self._cumulative_sizes[-1], \
        #         f"Max index {max(self._indices)} is greater than datasets " \
        #         f"list size {self._cumulative_sizes[-1]}"

    def _get_lazy_indices(self):
        """This method creates indices on-the-fly if self._indices=None.
        Only for internal use. Call may be expensive if self._indices=None.
        """
        if self._indices is not None:
            return self._indices
        else:
            return LazyRange(0, len(self))

    def subset(self: TFlatData, indices: Optional[Iterable[int]]) -> TFlatData:
        """Subsampling operation.

        :param indices: indices of the new samples
        :return:
        """
        if indices is not None and not isinstance(indices, list):
            indices = list(indices)

        if self._can_flatten and indices is not None:
            if self._indices is None:
                new_indices = indices
            else:
                self_indices = self._get_lazy_indices()
                new_indices = [self_indices[x] for x in indices]
            return self.__class__(datasets=self._datasets, indices=new_indices)
        return self.__class__(datasets=[self], indices=indices)

    def concat(self: TFlatData, other: TFlatData) -> TFlatData:
        """Concatenation operation.

        :param other: other dataset.
        :return:
        """
        if (not self._can_flatten) and (not other._can_flatten):
            return self.__class__(datasets=[self, other])

        # Case 1: one is a subset of the other
        if len(self._datasets) == 1 and len(other._datasets) == 1:
            if (
                self._can_flatten
                and self._datasets[0] is other
                and other._indices is None
            ):
                idxs = self._get_lazy_indices() + other._get_lazy_indices()
                return other.subset(idxs)
            elif (
                other._can_flatten
                and other._datasets[0] is self
                and self._indices is None
            ):
                idxs = self._get_lazy_indices() + other._get_lazy_indices()
                return self.subset(idxs)
            elif (
                self._can_flatten
                and other._can_flatten
                and self._datasets[0] is other._datasets[0]
            ):
                idxs = LazyIndices(self._get_lazy_indices(), other._get_lazy_indices())
                return self.__class__(datasets=self._datasets, indices=idxs)

        # Case 2: at least one of them can be flattened
        if self._can_flatten and other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                other_idxs = LazyIndices(other._get_lazy_indices(), offset=base_other)
                new_indices = self._get_lazy_indices() + other_idxs
            return self.__class__(
                datasets=self._datasets + other._datasets, indices=new_indices
            )
        elif self._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                if len(self._cumulative_sizes) == 0:
                    base_other = 0
                else:
                    base_other = self._cumulative_sizes[-1]
                other_idxs = LazyRange(0, len(other), offset=base_other)
                new_indices = self._get_lazy_indices() + other_idxs
            return self.__class__(
                datasets=self._datasets + [other], indices=new_indices
            )
        elif other._can_flatten:
            if self._indices is None and other._indices is None:
                new_indices = None
            else:
                base_other = len(self)
                self_idxs = LazyRange(0, len(self))
                other_idxs = LazyIndices(other._get_lazy_indices(), offset=base_other)
                new_indices = self_idxs + other_idxs
            return self.__class__(
                datasets=[self] + other._datasets, indices=new_indices
            )
        else:
            assert False, "should never get here"

    def _get_idx(self, idx) -> Tuple[int, int]:
        """Return the index as a tuple <dataset-idx, sample-idx>.

        The first index indicates the dataset to use from `self._datasets`,
        while the second is the index of the sample in
        `self._datasets[dataset-idx]`.

        Private method.
        """
        if self._indices is not None:  # subset indexing
            idx = self._indices[idx]
        if len(self._datasets) == 1:
            dataset_idx = 0
        else:  # concat indexing
            dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
            if dataset_idx == 0:
                idx = idx
            else:
                idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return dataset_idx, int(idx)

    @overload
    def __getitem__(self, item: int) -> T_co:
        ...

    @overload
    def __getitem__(self: TFlatData, item: slice) -> TFlatData:
        ...

    def __getitem__(self: TFlatData, item: Union[int, slice]) -> Union[T_co, TFlatData]:
        if isinstance(item, (int, np.integer)):
            dataset_idx, idx = self._get_idx(int(item))
            return self._datasets[dataset_idx][idx]
        else:
            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=item, max_length=len(self)
            )

            return self.subset(indices=slice_indices)

    def __len__(self) -> int:
        if len(self._cumulative_sizes) == 0:
            return 0
        elif self._indices is not None:
            return len(self._indices)
        return self._cumulative_sizes[-1]

    def __add__(self: TFlatData, other: TFlatData) -> TFlatData:
        return self.concat(other)

    def __radd__(self: TFlatData, other: TFlatData) -> TFlatData:
        return other.concat(self)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return _flatdata_repr(self)

    def _tree_depth(self):
        """Return the depth of the tree of datasets.
        Use only to debug performance issues.
        """
        return _flatdata_depth(self)


class ConstantSequence(IDataset[DataT], Sequence[DataT]):
    """A memory-efficient constant sequence."""

    def __init__(self, constant_value: DataT, size: int):
        """Constructor

        :param constant_value: the fixed value
        :param size: length of the sequence
        """
        self._constant_value = constant_value
        self._size = size
        self._can_flatten = False
        self._indices = None

    def __len__(self):
        return self._size

    @overload
    def __getitem__(self, index: int) -> DataT:
        ...

    @overload
    def __getitem__(self, index: slice) -> "ConstantSequence[DataT]":
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> "Union[DataT, ConstantSequence[DataT]]":
        if isinstance(index, (int, np.integer)):
            index = int(index)

            if index >= len(self):
                raise IndexError()
            return self._constant_value
        else:
            slice_indices = slice_alike_object_to_indices(
                slice_alike_object=index, max_length=len(self)
            )
            return ConstantSequence(
                constant_value=self._constant_value, size=sum(1 for _ in slice_indices)
            )

    def subset(self, indices: List[int]) -> "ConstantSequence[DataT]":
        """Subset

        :param indices: indices of the new data.
        :return:
        """
        return ConstantSequence(self._constant_value, len(indices))

    def concat(self, other: FlatData[DataT]) -> IDataset[DataT]:
        """Concatenation

        :param other: other dataset
        :return:
        """
        if (
            isinstance(other, ConstantSequence)
            and self._constant_value == other._constant_value
        ):
            return ConstantSequence(self._constant_value, len(self) + len(other))
        else:
            return FlatData([self, other])

    def __str__(self):
        return f"ConstantSequence(value={self._constant_value}, len={self._size})"

    def __hash__(self):
        return id(self)


def _flatten_dataset_list(
    datasets: List[Union[FlatData[T_co], IDataset[T_co]]]
) -> List[IDataset[T_co]]:
    """Flatten the dataset tree if possible."""
    # Concat -> Concat branch
    # Flattens by borrowing the list of concatenated datasets
    # from the original datasets.
    flattened_list: List[IDataset[T_co]] = []
    for dataset in datasets:
        if len(dataset) == 0:
            continue
        elif (
            isinstance(dataset, FlatData)
            and dataset._indices is None
            and dataset._can_flatten
        ):
            flattened_list.extend(dataset._datasets)
        else:
            flattened_list.append(dataset)

    # merge consecutive Subsets if compatible
    new_data_list: List[IDataset[T_co]] = []
    for dataset in flattened_list:
        last_dataset = new_data_list[-1] if len(new_data_list) > 0 else None
        if (
            isinstance(dataset, FlatData)
            and len(new_data_list) > 0
            and isinstance(last_dataset, FlatData)
        ):
            new_data_list.pop()
            merged_ds = _maybe_merge_subsets(last_dataset, dataset)
            new_data_list.extend(merged_ds)
        elif (
            (dataset is not None)
            and len(new_data_list) > 0
            and (last_dataset is not None)
            and last_dataset is dataset
        ):
            new_data_list.pop()
            # the same dataset is repeated, using indices to avoid repeating it
            idxs = LazyIndices(
                LazyRange(0, len(last_dataset)), LazyRange(0, len(last_dataset))
            )
            merged_ds = [FlatData([last_dataset], indices=idxs)]
            new_data_list.extend(merged_ds)
        else:
            new_data_list.append(dataset)
    return new_data_list


def _flatten_datasets_and_reindex(
    datasets: List[IDataset], indices: Optional[List[int]]
) -> Tuple[List[IDataset], Optional[List[int]]]:
    """The same dataset may occurr multiple times in the list of datasets.

    Here, we flatten the list of datasets and fix the indices to account for
    the new dataset position in the list.
    """
    # find unique datasets
    if not all(isinstance(d, Hashable) for d in datasets):
        return datasets, indices

    dset_uniques = set(datasets)
    if len(dset_uniques) == len(datasets):  # no duplicates. Nothing to do.
        return datasets, indices

    # split the indices into <dataset-id, sample-id> pairs
    cumulative_sizes = [0] + ConcatDataset.cumsum(datasets)
    data_sample_pairs: List[Tuple[int, int]] = []
    if indices is None:
        for ii, dset in enumerate(datasets):
            data_sample_pairs.extend([(ii, jj) for jj in range(len(dset))])
    else:
        for idx in indices:
            d_idx = bisect.bisect_right(cumulative_sizes, idx) - 1
            s_idx = idx - cumulative_sizes[d_idx]
            data_sample_pairs.append((d_idx, s_idx))

    # assign a new position in the list to each unique dataset
    new_datasets = list(dset_uniques)
    new_dpos = {d: i for i, d in enumerate(new_datasets)}
    new_cumsizes = [0] + ConcatDataset.cumsum(new_datasets)
    # reindex the indices to account for the new dataset position
    new_indices: List[int] = []
    for d_idx, s_idx in data_sample_pairs:
        new_d_idx = new_dpos[datasets[d_idx]]
        new_indices.append(new_cumsizes[new_d_idx] + s_idx)

    # NOTE: check disabled to avoid slowing down OCL scenarios
    # if len(new_indices) > 0 and new_cumsizes[-1] > 0:
    #     assert min(new_indices) >= 0
    #     assert max(new_indices) < new_cumsizes[-1], \
    #         f"Max index {max(new_indices)} is greater than datasets " \
    #         f"list size {new_cumsizes[-1]}"
    return new_datasets, new_indices


def _maybe_merge_subsets(d1: FlatData, d2: FlatData):
    """Check the conditions for merging subsets."""
    if (not d1._can_flatten) or (not d2._can_flatten):
        return [d1, d2]

    if (
        len(d1._datasets) == 1
        and len(d2._datasets) == 1
        and d1._datasets[0] is d2._datasets[0]
    ):
        # return [d1.__class__(d1._datasets, d1._indices + d2._indices)]
        return [d1.concat(d2)]
    return [d1, d2]


def _flatdata_depth(dataset):
    """Internal debugging method.
    Returns the depth of the dataset tree."""
    if isinstance(dataset, FlatData):
        dchilds = [_flatdata_depth(dd) for dd in dataset._datasets]
        if len(dchilds) == 0:
            return 1
        return 1 + max(dchilds)
    else:
        return 1


def _flatdata_print(dataset, indent=0):
    """Internal debugging method.
    Print the dataset."""
    print(_flatdata_repr(dataset, indent))


def _flatdata_repr(dataset, indent=0):
    """Return the string representation of the dataset.
    Shows the underlying dataset tree.
    """
    from avalanche.benchmarks.utils.data import _FlatDataWithTransform

    if isinstance(dataset, FlatData):
        ss = dataset._indices is not None
        cc = len(dataset._datasets) != 1
        cf = dataset._can_flatten
        s = (
            "\t" * indent
            + f"{dataset.__class__.__name__} (len={len(dataset)},subset={ss},"
            f"cat={cc},cf={cf})\n"
        )
        if isinstance(dataset, _FlatDataWithTransform):
            s = s[:-2] + (
                f",transform_groups={dataset._transform_groups},"
                f"frozen_transform_groups={dataset._frozen_transform_groups})\n"
            )
        for dd in dataset._datasets:
            s += _flatdata_repr(dd, indent + 1)
        return s
    else:
        return (
            "\t" * indent + f"{dataset.__class__.__name__} " f"(len={len(dataset)})\n"
        )


__all__ = ["FlatData", "ConstantSequence"]

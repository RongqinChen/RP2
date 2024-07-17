"""
Pytorch lightning data module for PyG dataset.
"""
import random
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Dataset

from .flat import FlatDataLoader
from .padded import PadDataLoader


class BatchSamplerWithGrouping():
    r"""Batch sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    """

    def __init__(self, sizes: List[int], batch_size: int, shuffle: bool, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don"t do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sizes = sizes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.__sample_groups = self._group_same_size()
        self.__batch_indices_list = None

    def _group_same_size(self):
        sample_groups = defaultdict(list)
        for idx, size in enumerate(self.sizes):
            sample_groups[size].append(idx)
        return sample_groups

    def _split_to_batches(self):
        index_list = []
        for group_indices in self.__sample_groups.values():
            if self.shuffle:
                random.shuffle(group_indices)
            index_list.extend(group_indices)

        batch_indices_list = [
            index_list[idx: idx + self.batch_size]
            for idx in range(0, len(index_list), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batch_indices_list)
        self.__batch_indices_list = batch_indices_list

    def __iter__(self) -> Iterator[List[int]]:
        if self.__batch_indices_list is None:
            self._split_to_batches()
        for batch_indices in self.__batch_indices_list:
            yield batch_indices
        self._split_to_batches()

    def __len__(self) -> int:
        if self.__batch_indices_list is None:
            self._split_to_batches()
        return len(self.__batch_indices_list)


class PlPyGDataModule(LightningDataModule):
    r"""Pytorch lightning data module for PyG dataset.
    Args:
        train_dataset (Dataset): Train PyG dataset.
        val_dataset (Dataset): Validation PyG dataset.
        test_dataset (Dataset): Test PyG dataset.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of process for data loader.
        drop_last (bool, optional): If true, drop the last batch during the training to avoid loss/metric inconsistence.
        pad2same (bool, optional): If flag is true, fill shapes so that all graphs are the same size, otherwise group graphs of the same size.
        follow_batch (list, optional): A list of key that will create a corresponding batch key in data loader.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 0,
        drop_last: Optional[bool] = False,
        pad2same: Optional[bool] = True,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        max_num_nodes: Optional[int] = None,
        train_sampler=None
    ):
        super(PlPyGDataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pad2same = pad2same
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.train_sampler = train_sampler
        if self.pad2same:
            if max_num_nodes is None:
                self.max_num_nodes = max(
                    [data.num_nodes for data in train_dataset] + [
                        data.num_nodes for data in val_dataset] + [
                        data.num_nodes for data in test_dataset]
                )
            else:
                self.max_num_nodes = max_num_nodes

    def _get_loader(self, split):
        if split == "train":
            dataset = self.train_dataset
            if self.train_sampler is not None:
                if self.pad2same:
                    return PadDataLoader(dataset, None, None, None, self.num_workers, None,
                                         batch_sampler=self.train_sampler, follow_batch=self.follow_batch,
                                         exclude_keys=self.exclude_keys)
                else:

                    return FlatDataLoader(dataset, 1, None, None, self.num_workers,
                                          batch_sampler=self.train_sampler, follow_batch=self.follow_batch,
                                          exclude_keys=self.exclude_keys)

        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset

        if self.pad2same:
            return PadDataLoader(dataset, self.batch_size, split == "train", self.drop_last,
                                 self.num_workers, self.max_num_nodes, follow_batch=self.follow_batch,
                                 exclude_keys=self.exclude_keys)
        else:
            return FlatDataLoader(dataset, self.batch_size, split == "train", self.drop_last,
                                  self.num_workers, follow_batch=self.follow_batch,
                                  exclude_keys=self.exclude_keys)

    def train_dataloader(self) -> DataLoader:
        return self._get_loader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_loader("val")

    def test_dataloader(self) -> DataLoader:
        return self._get_loader("test")


class PlPyGDataTestonValModule(PlPyGDataModule):
    r"""In validation mode, return both validation and test set for validation.
        Should use with PlGNNTestonValModule .
    """

    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return (super().val_dataloader(), super().test_dataloader())

    def test_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        return self.val_dataloader()

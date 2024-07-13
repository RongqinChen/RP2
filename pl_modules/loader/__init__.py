"""
Pytorch lightning data module for PyG dataset.
"""
from typing import Tuple, Optional, List
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
from .padded import PadDataLoader
from .flat import FlatDataLoader


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
        if self.pad2same:
            self.max_num_nodes = max(
                [data.num_nodes for data in train_dataset] + [
                    data.num_nodes for data in val_dataset] + [
                    data.num_nodes for data in test_dataset]
            )
        else:
            self.val_dataset = list(sorted(val_dataset, key=lambda data: data.num_nodes))
            self.test_dataset = list(sorted(test_dataset, key=lambda data: data.num_nodes))

    def _get_loader(self, split):
        if split == "train":
            dataset = self.train_dataset
        elif split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset

        if self.pad2same:
            return PadDataLoader(dataset, self.batch_size, split == "train", self.drop_last,
                                 self.num_workers, self.max_num_nodes, self.follow_batch, self.exclude_keys)
        else:
            return FlatDataLoader(dataset, self.batch_size, split == "train", self.drop_last,
                                  self.num_workers, self.follow_batch, self.exclude_keys)

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

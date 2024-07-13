from typing import Dict, List, Optional
import torch
import torch.utils.data
from torch_geometric.data import Data, Dataset


class FlatCollater:
    def __init__(
        self,
        dataset: Dataset,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Data]) -> Dict[str, torch.Tensor]:
        perm = sorted(range(len(batch)), key=lambda k: batch[k].num_nodes)
        batch = [batch[idx] for idx in perm]

        nn = [data.num_nodes for data in batch]
        en = [data.num_edges for data in batch]

        batch_num_nodes = torch.LongTensor([0] + nn)
        node_slice = torch.cumsum(batch_num_nodes, 0)
        node_batch = torch.repeat_interleave(batch_num_nodes[1:])

        batch_num_edges = torch.LongTensor([0] + en)

        eye_index = [
            torch.eye(n, dtype=torch.short).nonzero(as_tuple=False).t() for n in nn
        ]
        batch_eye_index = [findex + slice for findex, slice in zip(eye_index, node_slice)]
        batch_eye_index = torch.cat(batch_eye_index, 1)

        batch_node_val = [data.x for data in batch]
        batch_node_val = torch.cat(batch_node_val)

        batch_edge_val = [data.edge_attr for data in batch]
        batch_edge_val = torch.cat(batch_edge_val, 0).view((sum(en), -1))
        batch_edge_index = [data.edge_index + slice for data, slice in zip(batch, node_slice)]
        batch_edge_index = torch.cat(batch_edge_index, 1)

        pe_val = [data["pe_val"] for data in batch]
        batch_pe_val = torch.cat(pe_val, 0)
        batch_pe_index = [data["pe_index"] + slice for data, slice in zip(batch, node_slice)]
        batch_pe_index = torch.cat(batch_pe_index, 1)

        full_index = [
            torch.ones((n, n), dtype=torch.short).nonzero(as_tuple=False).t() for n in nn
        ]
        batch_full_index = [findex + slice for findex, slice in zip(full_index, node_slice)]
        batch_full_index = torch.cat(batch_full_index, 1)

        record = [[0, 0]]
        for n in nn:
            if record[-1][0] == n:
                record[-1][1] += 1
            else:
                record.append([n, 1])

        len1d = [cnt * n * n for n, cnt in record[1:]]
        size3d = [(cnt, n, n) for n, cnt in record[1:]]

        y = torch.cat([data.y for data in batch], 0)
        batch = {
            "batch_num_nodes": batch_num_nodes[1:], "batch_num_edges": batch_num_edges[1:],
            "total_num_nodes": sum(nn), "total_num_edges": sum(en),
            "node_batch": node_batch,
            "batch_eye_index": batch_eye_index, "batch_node_val": batch_node_val,
            "batch_edge_index": batch_edge_index, "batch_edge_val": batch_edge_val,
            "batch_pe_index": batch_pe_index, "batch_pe_val": batch_pe_val,
            "batch_full_index": batch_full_index,
            "len1d": len1d, "size3d": size3d, "y": y
        }
        return batch


class FlatDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
        num_workers: int = 1,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)
        kwargs.pop("shuffle", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=FlatCollater(dataset, follow_batch, exclude_keys),
            drop_last=drop_last,
            **kwargs,
        )

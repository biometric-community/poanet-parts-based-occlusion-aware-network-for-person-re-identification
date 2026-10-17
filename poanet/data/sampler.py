"""PK RandomIdentitySampler for batch-hard triplet."""

from __future__ import annotations

import collections
import random
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler[int]):
    def __init__(self, dataset, num_instances: int = 4, batch_size: int = 64):
        self.dataset = dataset
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_dic: Dict[int, List[int]] = collections.defaultdict(list)
        for i, lab in enumerate(dataset.labels):
            self.index_dic[int(lab)].append(i)
        self.pids = list(self.index_dic.keys())
        self.length = len(self.pids) // self.num_pids_per_batch * batch_size

    def __iter__(self) -> Iterator[int]:
        batch_idxs: List[int] = []
        pids = self.pids.copy()
        random.shuffle(pids)
        for pid in pids:
            idxs = self.index_dic[pid].copy()
            if len(idxs) < self.num_instances:
                idxs = random.choices(idxs, k=self.num_instances)
            else:
                random.shuffle(idxs)
                idxs = idxs[: self.num_instances]
            batch_idxs.extend(idxs)
            if len(batch_idxs) >= self.num_pids_per_batch * self.num_instances:
                yield from batch_idxs
                batch_idxs = []

    def __len__(self) -> int:
        return self.length

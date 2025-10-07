import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_metric_learning import samplers


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, use_sampler=True):
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
                'dataset': dataset,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'persistent_workers': True,
                'pin_memory': True,
                'prefetch_factor': 4
        }

        if shuffle and use_sampler:
            # Ensure every subject gets sampled (atleast) once per batch
            m = batch_size // dataset.num_subjects
            assert m != 0
            self.init_kwargs.update(
                **{'sampler': samplers.MPerClassSampler(
                    dataset.subject_ixs, m, batch_size=batch_size, length_before_new_iter=len(dataset.subject_ixs))}
            )
        else:
            self.init_kwargs.update(
                **{'shuffle': shuffle}
            )
        super().__init__(**self.init_kwargs)

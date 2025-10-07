from base import BaseDataLoader
from .datasets import *

class Sherlock(BaseDataLoader):
    def __init__(
        self, split: str, roi_name: str,
        batch_size: int, time_shuffle: bool, num_workers=5, use_sampler=True, shuffle=True, first_half=True):
        self.dataset = SherlockData(split, roi_name, time_shuffle, first_half=first_half)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            use_sampler=use_sampler)

class Forrest(BaseDataLoader):
    def __init__(
        self, split: str, roi_name: str,
        batch_size: int, time_shuffle: bool, num_workers=5, use_sampler=True, shuffle=True, first_half=True):
        self.dataset = ForrestData(split, roi_name, time_shuffle, first_half=first_half)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            use_sampler=use_sampler)

class fBIRNSubject(BaseDataLoader):
    def __init__(
        self, split: str, roi_name: str,
        batch_size: int, time_shuffle: bool, num_workers=5, use_sampler=True, shuffle=True):
        self.dataset = fBIRNSubjectData(split, roi_name, time_shuffle)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            use_sampler=False)

class SubjectMoons(BaseDataLoader):
    def __init__(
        self, split: str, roi_name: str,
        batch_size: int, time_shuffle: bool, num_workers=5, use_sampler=True, shuffle=True):
        self.dataset = SubjectMoonsData(split, roi_name, time_shuffle)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            use_sampler=False)

class Moons(BaseDataLoader):
    def __init__(
        self, split: str, roi_name: str,
        batch_size: int, time_shuffle: bool, num_workers=5, use_sampler=True, shuffle=True):
        self.dataset = MoonsData(split, roi_name, time_shuffle)
        super().__init__(
            dataset=self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            use_sampler=False)

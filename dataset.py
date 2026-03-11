import torch
from torch.utils.data import IterableDataset
from sampler import run as sampler_run

class SamplerDataset(IterableDataset):

    def __init__(self,args):
        self.args=args

    def __iter__(self):

        engine = sampler_run(self.args)

        for batch in engine:
            yield batch
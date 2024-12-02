import torch


class NNException(Exception):
    def __init__(self, nn: torch.nn.Module, *args: object):
        super().__init__(*args)
        self.nn = nn

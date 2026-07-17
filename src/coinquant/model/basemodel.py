import abc
import torch.nn as nn

class BaseModel(nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass
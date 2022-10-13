from autodiff import Parameter
from autodiff import Tensor

from typing import List


class Optimizer:
    params: List[Tensor]


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def optimize(self):
        for param in self.params:
            param.val = param.val - self.lr * param.gradient
        return self.params

    def zero_grad(self):
        for param in self.params:
            if isinstance(param, Parameter):
                param.zero_grad()

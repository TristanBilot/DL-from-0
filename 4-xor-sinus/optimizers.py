import numpy as np

from typing import List

from autodiff import Parameter
from autodiff import Tensor


class Optimizer:
    params: List[Tensor]

    def optimize(self):
        pass

    def zero_grad(self):
        for param in self.params:
            if isinstance(param, Parameter):
                param.zero_grad()


class SGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float=0.01,
        momentum: float=0.,
        nesterov: bool=False,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.state = [np.zeros_like(p.val) for p in self.params]

    def optimize(self):
        if self.momentum == 0.:
            for param in self.params:
                param.val = param.val - self.lr * param.gradient
        else:
            for i, p in enumerate(self.params):
                new_state = -self.lr * p.gradient + self.momentum * self.state[i]
                self.state[i] = new_state

            if self.nesterov:
                new_state = -self.lr * p.gradient + self.momentum * self.state[i]

            p.val = p.val + new_state


class Adam(Optimizer):
    class State:
        def __init__(self, mt: Tensor, vt: Tensor):
            self.mt = mt
            self.vt = vt

    def __init__(
        self,
        params: List[Tensor],
        lr: float=0.001,
        beta1: float=0.9,
        beta2: float=0.999,
        eps: float=1e-08,
    ):
        self.params = params
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.state = [Adam.State(
            mt=np.zeros(p.shape),
            vt=np.zeros(p.shape)) for p in self.params]
    
    def optimize(self):
        self.t += 1

        b1, b2 = self.beta1, self.beta2
        state, t = self.state, self.t

        for i, p in enumerate(self.params):
            state[i].mt = b1 * state[i].mt + (1 - b1) * p.gradient
            state[i].vt = b2 * state[i].vt + (1 - b2) * p.gradient ** 2.
            den = np.sqrt(state[i].vt) / np.sqrt((1 - b2 ** t)) + self.eps
            p.val = p.val - self.lr / (1 - b1 ** t) * (state[i].mt / den)

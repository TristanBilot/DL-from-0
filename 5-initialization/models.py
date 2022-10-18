from optimizers import Optimizer
from autodiff import Tensor
from layers import Layer, WeightedLayer


class MLP:
    def __init__(self, *layers) -> None:
        self.layers = layers

    def __call__(self, X: Tensor) -> Tensor:
        y_hat = X
        for layer in self.layers:
            y_hat = layer(y_hat)
        return y_hat

    def train(
        self,
        X: Tensor,
        y: Tensor,
        optimizer: Optimizer,
        loss_fn: Layer,
    ):
        y_hat = self(X)
        loss = loss_fn(y_hat, y)
        loss.backpropagate()
        optimizer.optimize()
        return loss, y_hat

    @property
    def params(self):
        params = []
        for l in self.layers:
            if isinstance(l, WeightedLayer):
                params += l.params
        return params

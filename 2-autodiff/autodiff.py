from typing import List, Union
import numpy as np


class Tensor:
    ValidInput = Union[float, list, np.ndarray]

    def __init__(self, val: ValidInput=None, a=None, b=None) -> None:
        self.val = self._force_np_array(val)
        self.a = a
        self.b = b
        self.gradient = np.zeros_like(self.val)
        self.tensor_type = "var"

    def __add__(self, other):
        tensor = Tensor(val=self.val + other.val, a=self, b=other)
        tensor.tensor_type = "add"
        return tensor

    def __sub__(self, other):
        tensor = Tensor(val=self.val - other.val, a=self, b=other)
        tensor.tensor_type = "sub"
        return tensor

    def __mul__(self, other):
        tensor = Tensor(val=self.val * other.val, a=self, b=other)
        tensor.tensor_type = "mul"
        return tensor

    def __truediv__(self, other):
        tensor = Tensor(val=self.val / other.val, a=self, b=other)
        tensor.tensor_type = "div"
        return tensor

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        tensor = Tensor(val=self.val ** other.val, a=self, b=other)
        tensor.tensor_type = "pow"
        return tensor

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        tensor = Tensor(val=np.dot(self.val, other.val), a=self, b=other)
        tensor.tensor_type = "matmul"
        return tensor

    def sigmoid(self, other):
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x))
        other = other if isinstance(other, Tensor) else Tensor(other)
        tensor = Tensor(val=_sigmoid(other.val), a=self, b=other)
        tensor.tensor_type = "sigmoid"
        return tensor

    def _force_np_array(self, val: ValidInput) -> np.ndarray:
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return val
        else:
            return np.asarray(val)

    def backpropagate(self, gradient=None):
        gradient = gradient if gradient is not None \
            else np.ones_like(self.val, dtype=np.float32)
        """ y = a
            dy/da = 1
        """
        if self.tensor_type == "var":
            self.gradient = self.gradient + gradient
        
        """ y = a + b
            dy/da = 1
            dy/db = 1
        """
        if self.tensor_type == "add":
            self.a.backpropagate(self._unbroadcast(gradient, self.a.shape))
            self.b.backpropagate(self._unbroadcast(gradient, self.b.shape))

        """ y = a - b
            dy/da = 1
            dy/db = -1
        """
        if self.tensor_type == "sub":
            self.a.backpropagate(self._unbroadcast(gradient, self.a.shape))
            self.b.backpropagate(self._unbroadcast(-gradient, self.b.shape))

        """ y = a * b
            dy/da = b
            dy/db = a
        """
        if self.tensor_type == "mul":
            self.a.backpropagate(self._unbroadcast(gradient * self.b, self.a.shape))
            self.b.backpropagate(self._unbroadcast(gradient * self.a, self.b.val.shape))

        """ y = a / b
            dy/da = 1 / b
            dy/db = -a / b ** 2
        """
        if self.tensor_type == "div":
            self.a.backpropagate(self._unbroadcast(gradient * 1 / self.b.val, self.a.shape))
            self.b.backpropagate(self._unbroadcast(gradient * -self.a.val / self.b.val ** 2, self.b.shape))

        """ y = a ** b
            dy/da = b * a ** b-1
            dy/db = a ** b * ln(a)
        """
        if self.tensor_type == "pow":
            self.a.backpropagate(gradient * self.b.val * self.a.val ** (self.b.val - 1))

        """ y = A @ B
            dy/dA = B.T
            dy/dB = A.T
        """
        if self.tensor_type == "matmul":
            self.a.backpropagate(np.dot(gradient, self.b.val.T))
            self.b.backpropagate(np.dot(self.a.val.T, gradient))

        """ y = sigmoid(a)
            dy/dA = sigmoid(a) * (1 - sigmoid(a))
        """
        if self.tensor_type == "sigmoid":
            def _sigmoid_derivative(x: float):
                return x * (1 - x)
            self.b.backpropagate(gradient * _sigmoid_derivative(self.val))


    # inspired from https://github.com/connor11son/stylegan-fmri/blob/66ccccbe081391bfd094ad94f5d9d9903115a1a8/torch_utils/ops/fma.py
    def _unbroadcast(self, x: np.ndarray, shape) -> np.ndarray:
        x = np.float32(x)
        extra_dims = x.ndim - len(shape)
        assert extra_dims >= 0
        dim = [i for i in range(x.ndim) if x.shape[i] > 1 and (i < extra_dims or shape[i - extra_dims] == 1)]
        if len(dim) != 0:
            dim = dim[0]
            x = x.sum(axis=dim, keepdims=True)
        if extra_dims:
            x = x.reshape(-1, *x.shape[extra_dims+1:])
        assert x.shape == shape
        return x

    @property
    def shape(self):
        return self.val.shape

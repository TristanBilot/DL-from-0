from typing import Union
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

    def __div__(self, other):
        tensor = Tensor(val=self.val / other.val, a=self, b=other)
        tensor.tensor_type = "div"
        return tensor

    def __pow__(self, other):
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
            else np.ones_like(self.val, dtype=np.float64)
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
            self.a.backpropagate(gradient)
            self.b.backpropagate(gradient)

        """ y = a - b
            dy/da = 1
            dy/db = -1
        """
        if self.tensor_type == "sub":
            self.a.backpropagate(gradient)
            self.b.backpropagate(-gradient)

        """ y = a * b
            dy/da = b
            dy/db = a
        """
        if self.tensor_type == "mul":
            self.a.backpropagate(gradient * self.b.val)
            self.b.backpropagate(gradient * self.a.val)

        """ y = a / b
            dy/da = 1 / b
            dy/db = -a / b ** 2
        """
        if self.tensor_type == "div":
            self.a.backpropagate(gradient * 1 / self.b.val)
            self.b.backpropagate(gradient * -self.a.val / self.b.val ** 2)

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
            self.b.backpropagate(gradient * _sigmoid_derivative(self.val))

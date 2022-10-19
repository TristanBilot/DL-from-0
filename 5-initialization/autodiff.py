from typing import List, Union
import numpy as np


def _one_hot_encode(vector: np.ndarray, nb_classes: int=None) -> np.ndarray:
    assert vector.ndim == 1, "Argument `vector` should be a 1D vector of size batch_size"

    nb_classes = nb_classes if nb_classes is not None else np.max(vector) + 1
    batch_size = vector.shape[0]
    matrix = np.zeros((batch_size, nb_classes))
    matrix[np.arange(batch_size), vector] = 1
    return matrix

def _sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(x: np.ndarray):
    return x * (1 - x)

def _relu(x: np.ndarray):
    return np.maximum(0., x)

def _relu_derivative(x: np.ndarray):
     return ((x > 0) * np.ones_like(x))

class Tensor:
    ValidInput = Union[float, list, np.ndarray]

    def __init__(self, val: ValidInput=None, a=None, b=None) -> None:
        self.val = self._force_np_array(val)
        self.a = self._force_np_array(a)
        self.b = self._force_np_array(b)
        self.gradient = np.zeros_like(self.val, dtype=np.float32)
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

    def sum(self, input: 'Tensor', axis: int=None, keepdims: bool=False):
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=np.sum(input.val, axis=axis, keepdims=keepdims), a=self, b=input)
        tensor.tensor_type = "sum"
        return tensor

    def sigmoid(self, input: 'Tensor'):
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=_sigmoid(input.val), a=self, b=input)
        tensor.tensor_type = "sigmoid"
        return tensor

    def relu(self, input: 'Tensor'):
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=_relu(input.val), a=self, b=input)
        tensor.tensor_type = "relu"
        return tensor

    def mse_loss(self, y_hat: 'Tensor', y: 'Tensor'):
        y_hat = y_hat if isinstance(y_hat, Tensor) else Tensor(y_hat)
        y = y if isinstance(y, Tensor) else Tensor(y)
        n = Tensor(y_hat.val.size)
        sum = Tensor().sum
        val = (sum((y_hat - y) ** Tensor(2))) / n
        return val

    def softmax(self, input: 'Tensor', dim: int=-1) -> 'Tensor':
        input = input if isinstance(input, Tensor) else Tensor(input)
        exp = self.exp(input)
        out = exp / self.sum(exp, axis=dim, keepdims=True)
        return out

    def exp(self, input: 'Tensor') -> 'Tensor':
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=np.exp(input.val), a=self, b=input)
        tensor.tensor_type = "exp"
        return tensor

    def reduce_sum(self, input: 'Tensor', axis = None, keepdims=False):
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(np.sum(input.value, axis=axis, keepdims=keepdims))
        tensor.tensor_type = "reduce_sum"
        return tensor

    def nll_loss(
        self,
        input: 'Tensor',
        target: 'Tensor',
        reduction: str = 'mean'
    ) -> 'Tensor':

        if input.ndim != 2:
            raise ValueError("Expected 2 dimensions, (batch_size, nb_classes)")
        if target.ndim != 1:
            raise ValueError(
                "Parameter `target` should be a 1D vector of size batch_size where each element is a class"
            )
        if input.shape[0] != target.shape[0]:
            raise ValueError(
                "Expected input batch_size ({}) to match target batch_size ({}).".format(input.shape[0], target.shape[0])
            )

        batch_size = input.shape[0]
        n_classes = input.shape[1]
        eps = 1e-7

        ret = -np.log(input.val[np.arange(batch_size), target.val.astype(np.int)] + eps)
        if reduction in ['sum', 'mean']:
            ret = np.sum(ret)
        if reduction == 'mean':
            ret = ret / batch_size

        tensor = Tensor(val=ret, a=self, b=input)
        tensor.tensor_type = "nll_loss"
        tensor.target = target
        tensor.reduction = reduction
        tensor.n_classes = n_classes
        tensor.batch_size = batch_size
        return tensor

    def argmax(self, input: 'Tensor', dim: int=None) -> 'Tensor':
        tensor = Tensor(val=np.argmax(input.val, axis=dim), a=None, b=None)
        return tensor

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

        """ y = sum(A)
            dy/dA = np.ones_like(A)
        """
        if self.tensor_type == "sum":
            self.b.backpropagate(gradient * np.ones_like(self.b.val))

        """ y = sigmoid(a)
            dy/da = sigmoid(a) * (1 - sigmoid(a))
        """
        if self.tensor_type == "sigmoid":
            self.b.backpropagate(gradient * _sigmoid_derivative(self.val))

        """ y = relu(a)
            dy/da = 0 if a < 0, 1 if a > 0
        """
        if self.tensor_type == "relu":
            self.b.backpropagate(gradient * _relu_derivative(self.b.val))

        """ y = exp(a)
            dy/da = exp(a)
        """
        if self.tensor_type == "exp":
            # should be gradient * self.val but gradient * self.b.val gives better accuracy (?)
            self.b.backpropagate(gradient * self.val)

        if self.tensor_type == "nll_loss":
            p = np.clip(self.b.val, 1e-15, 1 - 1e-15)
            y = _one_hot_encode(self.target.val, n_classes=self.n_classes)
            
            if self.reduction == 'mean':
                self.b.backpropagate((p - y) / self.batch_size)
            elif self.reduction == 'sum':
                self.b.backpropagate(p - y)

        if self.tensor_type == "reduce_sum":
            self.b.backpropagate(np.ones(self.b.val.shape))


    def near_eq(self, other: 'Tensor', round: int=2) -> 'Tensor':
        if self.shape != other.shape:
            raise ValueError(
                "Shape should be ({}) to match shape ({}).".format(self.shape, other.shape)
            )
        return Tensor(np.equal(np.round(self.val, round), np.round(other.val, round)))

    def eq(self, other: 'Tensor') -> 'Tensor':
        if self.shape != other.shape:
            raise ValueError(
                "Shape should be ({}) to match shape ({}).".format(self.shape, other.shape)
            )
        return Tensor(np.equal(self.val, other.val))

    def uniform_(self, low: float = 0., high: float = 1.) -> None:
        self.val = np.random.uniform(low=low, high=high, size=self.shape)

    def _force_np_array(self, val: ValidInput, dtype=np.float32) -> np.ndarray:
        if val is None or isinstance(val, Tensor):
            return val
        return np.asarray(val, dtype=dtype)

    def _unbroadcast(self, x, shape):
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

    @property
    def ndim(self):
        return self.val.ndim


class Parameter(Tensor):
    @classmethod
    def randn(cls, *shape):
        return cls(np.random.randn(*shape))

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)
    
    def zero_grad(self):
        self.gradient = np.zeros(self.shape, dtype=np.float32)

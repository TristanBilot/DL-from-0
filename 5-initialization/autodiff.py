from typing import List, Union
import numpy as np


def to_categorical(target: np.ndarray, n_classes: int = None) -> np.ndarray:
	"""
	Convert a class vector (integers) to binary class matrix.

	Parameters
    ----------
    target : np.ndarray
        A 1-dim (batch_size) class vector to be converted into a matrix
        (integers from 0 to n_classes - 1).

    n_classes : int, optional
        Total number of classes. If `None`, this would be inferred as the
        (largest number in target) + 1.

	Returns
    -------
	one_hot : np.ndarray
        A binary class matrix (batch_size, n_classes)
	"""
	n_classes = n_classes if n_classes is not None else np.max(target) + 1
	batch_size = target.shape[0]
	one_hot = np.zeros((batch_size, n_classes))
	one_hot[np.arange(batch_size), target] = 1
	return one_hot

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
        # out_grad = out.grad
        #         if out.ndim < self.ndim:
        #             sum_dim = [dim] if type(dim) is int else dim
        #             expanded_shape = [1 if sum_dim is None or i in sum_dim else self.shape[i] for i in range(len(self.shape))]
        #             out_grad = out_grad.reshape(expanded_shape)
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
        out = input - self.max(input, dim=dim, keepdims=True)
        out = self.exp(out)
        out = out / self.sum(out, axis=dim, keepdims=True)
        return out

    def max(self, input: 'Tensor', dim: int = None, keepdims: bool = False) -> 'Tensor':
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=np.max(input.val, axis=dim, keepdims=keepdims), a=self, b=input)
        tensor.tensor_type = "max"
        tensor.dim = dim
        return tensor

    def exp(self, input: 'Tensor') -> 'Tensor':
        input = input if isinstance(input, Tensor) else Tensor(input)
        tensor = Tensor(val=np.exp(input.val), a=self, b=input)
        tensor.tensor_type = "exp"
        return tensor

    def nll_loss(
        self,
        input: 'Tensor',
        target: 'Tensor',
        reduction: str = 'mean'
    ) -> 'Tensor':
        """
        Negative Log Likelihood Loss

        NOTE:
            Here I apply ``log()`` on the prediction data, which is DIFFERENT
            FROM ``nn.functional.nll_loss()`` IN PYTORCH!

        Parameters
        ----------
        input : Tensor
            A 2-dim (batch_size, n_classes) tensor

        target : Tensor
            A 1-dim (batch_size) tensor where each value: 0 <= target[i] <= n_classes-1

        reduction : str, optional, default='mean'
            'none' / 'mean' / 'sum'
        """
        dim = input.ndim

        if dim != 2:
            raise ValueError("Expected 2 dimensions (got {})".format(dim))

        if input.shape[0] != target.shape[0]:
            raise ValueError(
                "Expected input batch_size ({}) to match target batch_size ({}).".format(input.shape[0], target.shape[0])
            )

        batch_size = input.shape[0]
        n_classes = input.shape[1]
        delta = 1e-7  # deal with the situation that input.data = 0

        ret = - np.log(input.val[np.arange(batch_size), target.val.astype(np.int)] + delta)
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

    def __repr__(self) -> str:
        return self.val
        if self.tensor_type == "var":
            return f"{self.val}"
        
        if self.tensor_type == "add":
            return f"{self.a} + {self.b}"

        if self.tensor_type == "sub":
            return f"{self.a} - {self.b}"

        if self.tensor_type == "mul":
            return f"({self.a} * {self.b})"

        if self.tensor_type == "div":
            return f"({self.a} / {self.b})"

        if self.tensor_type == "pow":
            return f"({self.a} ^ {self.b})"

        if self.tensor_type == "matmul":
            return f"({self.a} . {self.b})"

        if self.tensor_type == "sigmoid":
            return f"sig({self.b})"

        if self.tensor_type == "mse_loss":
            return f"mse_loss({self.b})"

        if self.tensor_type == "relu":
            return f"relu({self.b})"

        if self.tensor_type == "sum":
            return f"sum({self.b})"

    def backpropagate(self, gradient=None):
        gradient = gradient if gradient is not None \
            else np.ones_like(self.val, dtype=np.float32)
        """ y = a
            dy/da = 1
        """
        if self.tensor_type == "var":
            self.gradient = self.gradient + gradient
            # self.gradient = self._unbroadcast_addition(self.gradient, gradient)
        
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
            dy/dA = sigmoid(a) * (1 - sigmoid(a))
        """
        if self.tensor_type == "sigmoid":
            self.b.backpropagate(gradient * _sigmoid_derivative(self.val))

        """ y = relu(a)
            dy/dA = 0 if a < 0, 1 if a > 0
        """
        if self.tensor_type == "relu":
            self.b.backpropagate(gradient * _relu_derivative(self.b.val))


        if self.tensor_type == "max":
            out_grad = self.gradient
            out_data = self.val
            dim = self.dim
            if self.ndim < self.b.ndim:
                max_dim = [dim] if type(dim) is int else dim
                # here I don't use np.expand_dims(), because I have to deal
                # with the situation when ``dim = None```
                expanded_shape = [1 if max_dim is None or i in max_dim else self.b.shape[i] for i in range(len(self.b.shape))]
                out_grad = out_grad.reshape(expanded_shape)
                out_data = out_data.reshape(expanded_shape)
            mask = (self.b.val == out_data)
            self.b.gradient += mask * out_grad

        if self.tensor_type == "exp":
            self.b.backpropagate(gradient * self.b.val)
            # out.grad * out.data

        if self.tensor_type == "nll_loss":
            # if self.reduction != 'none':
            p = np.clip(self.b.val, 1e-15, 1 - 1e-15)
            y = to_categorical(self.target.val, n_classes=self.n_classes)
            if self.reduction == 'mean':
                self.b.backpropagate((p - y) / self.batch_size)  # (batch_size, n_classes)
            elif self.reduction == 'sum':
                self.b.backpropagate(p - y)  # (batch_size, n_classes)


    def near_eq(self, other: 'Tensor', round: int=2) -> 'Tensor':
        if self.shape != other.shape:
            raise ValueError(
                "Shape should be ({}) to match shape ({}).".format(self.shape, other.shape)
            )
        return Tensor(np.equal(np.round(self.val, round), np.round(other.val, round)))

    def uniform_(self, low: float = 0., high: float = 1.) -> None:
        """
        Fill the tensor with values drawn from the uniform distribution.

        Parameters
        ----------
        low : float, optional, default=0.
            The lower bound of the uniform distribution

        high : float, optional, default=1.
            The upper bound of the uniform distribution
        """
        self.val = np.random.uniform(low=low, high=high, size=self.shape)


    def _force_np_array(self, val: ValidInput, dtype=np.float32) -> np.ndarray:
        if val is None or isinstance(val, Tensor):
            return val

        # if isinstance(val, np.ndarray):
        #     if val.ndim == 1:
        #         # A valid matrix is at least of shape (n, m), not (n,)
        #         val = val.reshape(-1, 1)

        return np.asarray(val, dtype=dtype)

    def _unbroadcast_addition(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # if (type(a) == int or type(b) == int) or self.can_be_broadcast(a, b) :
        #     return a + b
        unmatched_axis = [i for i, s in enumerate(b.shape) if s != a.shape[i]]
        for axis in unmatched_axis:
            b = b.sum(axis=axis, keepdims=True)
        return a + b

    
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

        
if __name__ == '__main__':
    # W1 = Tensor(np.array([[-0.5,  0.5], [-2,  2]]))
    # b1 = Tensor(np.array([[1.2,  1.2], [1.2,  1.2]]))
    # W2 = Tensor(np.array([[-1.5,  1.5], [-2,  2]]))
    # b2 = Tensor(np.array([[0.6,  0.7], [1.2,  1.2]]))

    # X = Tensor(np.array([[-2,  2], [1.5, 1.5]]))
    # sig = Tensor().sigmoid
    # mse = Tensor().mse_loss

    # c = mse(W1, W2)
    # # c = sig(W2 @ sig(W1 @ X + b1) + b2)
    # c.backpropagate()

    # print(f'{W2.gradient}')

    a = np.array([[-0.5], [-2]])
    b = np.array([[1.2], [1.2]])
    y_hat = Tensor(a)
    y = Tensor(b)

    mse = Tensor().mse_loss
    c = mse(y_hat, y)
    c.backpropagate()

    print(f'{y_hat.gradient}')

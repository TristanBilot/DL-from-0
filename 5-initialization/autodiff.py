from typing import List, Union
import numpy as np

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(x: float):
    return x * (1 - x)

def _relu(x: float):
    return np.maximum(0., x)

def _relu_derivative(x: float):
     return ((x > 0) * np.ones_like(x))

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
            self.gradient = self._unbroadcast_addition(self.gradient, gradient)
        
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
            self.b.backpropagate(gradient * _relu_derivative(self.val))


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
        if (type(a) == int or type(b) == int) or (a.ndim == 0 or b.ndim == 0):
            return a + b

        longest = a if len(a.shape) >= len(b.shape) else b
        shortest = a if len(a.shape) < len(b.shape) else b

        additional_axis = [i for i, s in enumerate(shortest.shape) if s != longest.shape[i]]
        for axis in additional_axis:
            shortest = shortest.sum(axis=axis, keepdims=True)
        return longest + shortest

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

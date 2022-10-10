# Chain rule from scracth

Simple engine to compute composite functions derivatives using the chain rule. The aim of this repo is to understand easily how chain rule works and how derivatives are computed automatically using a computation graph. This is a first step towards automtic differentiation.

Only `cos`, `sin`, `exp` and `pow` operations are implemented but additional ones could be easily added but require some parsing boilerplate code to develop.

## Derivative computation

```python
exp = "sin(exp(sin(5x^9)))"
derivative = Expression(exp).differentiate()

derivative
# >>> "cos(exp(sin(5x^9))) * exp(sin(5x^9)) * cos(5x^9) * 45x^8"
```

## Derivative inference

```python
exp = "cos(3x^2)"
derivative = Expression(exp).differentiate(x=5)

derivative
# >>> 11.6334490623
```

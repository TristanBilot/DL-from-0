import math


FUNCS = ["cos", "sin", "exp", "-cos", "-sin", "-exp", "*"]

class Node:
    def value(self): ...
    def derivative(self): ...

    def __repr__(self):
        return self.val


class ExprNode(Node):
    def __init__(self, val):
        self.val = val

    def value(self):
        return {
            "infer": lambda x: self._make_calculus(self.val, x),
            "string": self.val
        }

    def derivative(self):
        constant, var, power = self._get_constant_var_pow(self.val)
        if power == "":
            return {
            "infer": lambda x: self._make_calculus(self.val, x),
            "string": constant,
        }
        power = int(power)
        constant = int(constant if constant != "" else 1)
        expr = f"{power * constant}{var}{'^' + str(power - 1) if power != 2 else ''}"

        return {
            "infer": lambda x: self._make_calculus(expr, x),
            "string": expr
        }

    def _get_constant_var_pow(self, expr):
        reach_pow = False
        buf, constant, var, power = "", "", "", ""
        for char in expr:
            if not char.isnumeric() and char != "^":
                var = char
                constant = buf
            else:
                buf += char
            if char == "^":
                reach_pow = True
            elif reach_pow:
                power += char

        return constant, var, power

    def _make_calculus(self, expr, x):
        constant, var, power = self._get_constant_var_pow(expr)
        constant = int(constant) if constant != "" else 1
        power = int(power) if power != "" else 1

        return constant * (x ** power)



class FuncNode(Node):
    def __init__(self, val):
        self.val = val

    def value(self):
        funcs = {
            "cos": {
                "infer": lambda x: math.cos(x),
                "string": "cos"
            },
            "sin": {
                "infer": lambda x: math.sin(x),
                "string": "sin"
            },
            "exp": {
                "infer": lambda x: math.exp(x),
                "string": "exp"
            },
            "-cos": {
                "infer": lambda x: -math.cos(x),
                "string": "-cos"
            },
            "-sin": {
                "infer": lambda x: -math.sin(x),
                "string": "-sin"
            },
            "-exp": {
                "infer": lambda x: -math.exp(x),
                "string": "-exp"
            },
            "mul": {
                "infer": lambda x, y: x * y,
                "string": "*"
            }
        }
        return funcs[self.val]

    def derivative(self):
        derivatives = {
            "cos": {
                "infer": lambda x: -math.sin(x),
                "string": "-sin"
            },
            "sin": {
                "infer": lambda x: math.cos(x),
                "string": "cos"
            },
            "exp": {
                "infer": lambda x: math.exp(x),
                "string": "exp"
            },
            "-cos": {
                "infer": lambda x: math.sin(x),
                "string": "sin"
            },
            "-sin": {
                "infer": lambda x: -math.cos(x),
                "string": "-cos"
            },
            "-exp": {
                "infer": lambda x: -math.exp(x),
                "string": "-exp"
            },
        }
        return derivatives[self.val]

        
class Parser:
    def __init__(self, expr):
        self.expr = expr

    def parse(self):
        queue = []
        idx, buf = 0, ""
        for i, char in enumerate(self.expr):
            if char == "(" or (char == ")" and self.expr[i-1] != ")"):
                queue.append(buf)
                buf = ""
            else:
                if char in [")", " "]:
                    continue
                buf += char

                if char == "*" or (i == len(self.expr) - 1 and char != ")"):
                    queue.append(buf)
                    buf = ""

        return [self._forge_node(x) for x in queue]

    def _forge_node(self, buf: str):
        if buf in FUNCS:
            return FuncNode(buf)
        return ExprNode(buf)


class Expression:
    def __init__(self, expr):
        self.parser = Parser(expr)
        self.nodes = self.parser.parse()

    def differentiate(self, x=None):
        if x is None:
            return self._differentiate()
        return self._infer(x=x)

    def _differentiate(self):
        expr = ""
        nodes = self.nodes

        for i, derivative_node in enumerate(nodes):
            expr += self._update_expr(derivative_node, derivative=True, field="string")
            for no_derivative_node in nodes[i+1:]:
                expr += self._update_expr(no_derivative_node, derivative=False, field="string")

            nb_funcs = len([n for n in nodes[i+1:] \
                if isinstance(derivative_node, FuncNode)])
            expr += ')' * nb_funcs
            expr += " * " if i != len(nodes) - 1 else ""
        return expr

    def _infer(self, x):
        derivative = self._differentiate()
        nodes = Parser(derivative).parse()

        mul_groups, tmp = [], []
        for node in nodes:
            if node.val == "*":
                mul_groups.append(tmp)
                tmp = []
            else:
                tmp.append(node)
        mul_groups.append(tmp)

        product = 1
        for group in mul_groups:
            tmp = x
            for node in reversed(group):
                func = node.value()["infer"]
                tmp = func(x=tmp)
            product *= tmp

        return product


    def _update_expr(self, node, derivative: bool, field: str):
        if derivative:
            expr = node.derivative()[field]
        else:
            expr = node.value()[field]
        if isinstance(node, FuncNode):
            expr += "("
        return expr
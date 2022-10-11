import math

from chain_rule import *


def test_expr_node1():
    node = ExprNode("2x")
    assert node.value()["string"] == "2x"
    assert node.derivative()["string"] == "2"

def test_expr_node2():
    node = ExprNode("x^2")
    assert node.derivative()["string"] == "2x"

def test_expr_node3():
    node = ExprNode("2x^2")
    assert node.derivative()["string"] == "4x"

def test_expr_node4():
    node = ExprNode("3x^10")
    assert node.derivative()["string"] == "30x^9"

def test_expr_node5():
    node = ExprNode("10x^11")
    assert node.derivative()["string"] == "110x^10"


def test_Expression1():
    exp = "cos(sin(2x^2))"
    diff = Expression(exp)
    assert diff.differentiate() == "-sin(sin(2x^2)) * cos(2x^2) * 4x"

def test_Expression2():
    exp = "sin(exp(sin(x^9)))"
    diff = Expression(exp)
    assert diff.differentiate() == "cos(exp(sin(x^9))) * exp(sin(x^9)) * cos(x^9) * 9x^8"

def test_Expression3():
    exp = "cos(cos(sin(exp(sin(4x)))))"
    diff = Expression(exp)
    assert diff.differentiate() == "-sin(cos(sin(exp(sin(4x))))) * -sin(sin(exp(sin(4x)))) * cos(exp(sin(4x))) * exp(sin(4x)) * cos(4x) * 4"


def test_inference1():
    exp = "sin(exp(sin(x^9)))"
    diff = Expression(exp)
    assert math.ceil(diff.differentiate(x=3)) == -14475

def test_inference2():
    exp = "cos(2x)"
    diff = Expression(exp)
    assert round(diff.differentiate(x=2), 2) == 1.51

def test_inference3():
    exp = "cos(exp(sin(cos(cos(3x^9)))))"
    diff = Expression(exp)
    assert round(diff.differentiate(x=5), 2) == 423175.18
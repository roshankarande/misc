"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy 

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a,self.scalar)

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a/b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return b**-1, -a/(b)**2
        # return power_scalar(b,-1), multiply(negate(a), power_scalar(b,-2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return  out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None ):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = (len(a.shape) - 1), (len(a.shape) - 2)
            # self.axes = a.shape[-1] , a.shape[-2]
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        # a = node.inputs[0]
        # return array_api.swapaxes(out_grad.numpy(), *a.axes)
        return out_grad


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # return out_grad.reshape(self.shape)
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
# ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        if len(lhs.shape) > len(rhs.shape):
            axis = tuple(i for i in range(len(lhs.shape) - len(rhs.shape)))
            return out_grad @ rhs.transpose(), (lhs.transpose() @ out_grad).sum(axes=axis)

        if len(rhs.shape) > len(lhs.shape):
            axis = tuple(i for i in range(len(rhs.shape) - len(lhs.shape)))
            da1 = summation(out_grad @ rhs.transpose(), axes=axis)
            da2 = lhs.transpose() @ out_grad
            return da1, da2
        
        return out_grad @ rhs.transpose(), lhs.transpose() @ out_grad



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1*a

    def gradient(self, out_grad, node):
        return -1*out_grad



def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):   
        return array_api.log(a)

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)



class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a);
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a,0)

    def gradient(self, out_grad, node):
        ### # TODO: BEGIN YOUR SOLUTION
        pass
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


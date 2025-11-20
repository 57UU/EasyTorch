from typing import Dict
import enum
import math

class OpType(enum.Enum):
    Add=1
    Sub=2
    Mul=3
    Div=4
    Sigmoid=5
    Log=7
    Exp=8

class GradNumber:
    ZERO:'GradNumber'
    number_cache:Dict[float,'GradNumber']={}
    @staticmethod
    def get_number_no_grad(value:float)->'GradNumber':
        tryGet=GradNumber.number_cache.get(value,None)
        if tryGet is not None:
            return tryGet
        instance=GradNumber(value,requires_grad=False)
        GradNumber.number_cache[value]=instance
        return instance
    def __init__(self, value, requires_grad=True):
        # if not isinstance(value, (int, float)):
        #     raise ValueError("GradNumber value must be a number")
        self.value = value
        self.grad = 0
        self.requires_grad = requires_grad
        self.preOp1:GradNumber=None
        self.preOp2:GradNumber=None
        self.opType:OpType=None
    # def __setattr__(self, name: str, value) -> None:
    #     if name=="value":
    #         if isinstance(value, GradNumber):
    #             raise ValueError("GradNumber cannot be nested")
    #     super().__setattr__(name, value)
    def backward(self):
        if not self.requires_grad:
            return
        self.grad=1
        self._backward()
    def _backward(self):
        if self.opType is None or not self.requires_grad:
            return
        if self.opType==OpType.Add:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad
            if self.preOp2.requires_grad:
                self.preOp2.grad+=self.grad
        elif self.opType==OpType.Sub:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad
            if self.preOp2.requires_grad:
                self.preOp2.grad+=self.grad*-1
        elif self.opType==OpType.Mul:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad*self.preOp2.value
            if self.preOp2.requires_grad:
                self.preOp2.grad+=self.grad*self.preOp1.value
        elif self.opType==OpType.Div:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad/self.preOp2.value
            if self.preOp2.requires_grad:
                self.preOp2.grad+=self.grad*self.preOp1.value*-1/self.preOp2.value**2
        elif self.opType==OpType.Sigmoid:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad*self.value*(1-self.value)
        elif self.opType==OpType.Log:
            if self.preOp1.requires_grad:
                self.preOp1.grad+=self.grad/self.preOp1.value
        elif self.opType==OpType.Exp:
            if self.preOp1.requires_grad:
                #eq to self.grad*math.exp(self.preOp1.value)
                self.preOp1.grad+=self.grad*self.value
        else:
            raise ValueError("Unknown op type")
        if self.preOp1 is not None:
            self.preOp1._backward()
        if self.preOp2 is not None:
            self.preOp2._backward()

    def __add__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        ret = GradNumber(self.value + other.value, self.requires_grad or other.requires_grad)
        ret.preOp1 = self
        ret.preOp2 = other
        ret.opType = OpType.Add
        return ret
    def __sub__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        ret = GradNumber(self.value - other.value, self.requires_grad or other.requires_grad)
        ret.preOp1 = self
        ret.preOp2 = other
        ret.opType = OpType.Sub
        return ret
    def __mul__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        ret = GradNumber(self.value * other.value, self.requires_grad or other.requires_grad)
        ret.preOp1 = self
        ret.preOp2 = other
        ret.opType = OpType.Mul
        return ret
    
    def __truediv__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        ret = GradNumber(self.value / other.value, self.requires_grad or other.requires_grad)
        ret.preOp1 = self
        ret.preOp2 = other
        ret.opType = OpType.Div
        return ret
    
    def sigmoid(x):
        if not isinstance(x, GradNumber):
            x = GradNumber.get_number_no_grad(x)
        #  1/(1+e^-x)
        value = 1 / (1 + math.exp(-x.value))
        ret = GradNumber(value, x.requires_grad)
        ret.preOp1 = x
        ret.preOp2 = None
        ret.opType = OpType.Sigmoid
        return ret
    
    
    def log(x):
        if not isinstance(x, GradNumber):
            x = GradNumber.get_number_no_grad(x)
        # 计算自然对数
        value = math.log(x.value)
        ret = GradNumber(value, x.requires_grad)
        ret.preOp1 = x
        ret.preOp2 = None
        ret.opType = OpType.Log
        return ret
    
    def exp(x):
        if not isinstance(x, GradNumber):
            x = GradNumber.get_number_no_grad(x)
        # 计算指数函数
        value = math.exp(x.value)
        ret = GradNumber(value, x.requires_grad)
        ret.preOp1 = x
        ret.preOp2 = None
        ret.opType = OpType.Exp
        return ret
    
    # 反向操作魔法方法
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        return other.__sub__( self)
    
    def __rmul__(self, other):
        # 乘法满足交换律，所以可以复用__mul__
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        if not isinstance(other, GradNumber):
            other = GradNumber.get_number_no_grad(other)
        return other.__truediv__(self)
    # --- comparison support ---
    def __lt__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value <= other

    def __eq__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value != other

    def __gt__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, GradNumber):
            other = other.value
        return self.value >= other

GradNumber.ZERO=GradNumber(0,False)
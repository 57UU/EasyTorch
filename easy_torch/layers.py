from .module import Module
from .tensor import Tensor
from .init import xavier_uniform_

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.zeros((in_features, out_features),requires_grad=True)
        self.bias = Tensor.randn((out_features,),requires_grad=True)
        xavier_uniform_(self.weight)

    def forward(self, x: Tensor):
        return  x @ self.weight + self.bias

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: Tensor):
        """
        计算softmax函数
        Args:
            x: 输入张量
            dim: 计算softmax的维度，默认为最后一个维度
        Returns:
            softmax后的张量
        """
        import math
        
        # 处理负维度
        if self.dim < 0:
            self.dim = x.ndim + self.dim
        
        # 确保维度有效
        if self.dim < 0 or self.dim >= x.ndim:
            raise ValueError(f"Dimension {self.dim} is out of bounds for tensor of dimension {x.ndim}")
        
        # 创建结果张量
        result = Tensor(x.shape, x.requires_grad)
        
        # 遍历所有可能的索引
        for idx in x._get_indices():
            # 计算当前索引在指定维度上的所有可能值
            # 获取除了指定维度外的所有索引
            other_dims = list(idx)
            other_dims[self.dim] = slice(None)  # 使用切片表示所有可能值
            
            # 找到最大值以提高数值稳定性
            max_val = x[idx].value
            for i in range(x.shape[self.dim]):
                temp_idx = list(idx)
                temp_idx[self.dim] = i
                val = x[tuple(temp_idx)].value
                if val > max_val:
                    max_val = val
            
            # 计算指数和
            exp_sum = 0.0
            exp_values = []
            for i in range(x.shape[self.dim]):
                temp_idx = list(idx)
                temp_idx[self.dim] = i
                val = x[tuple(temp_idx)].value
                exp_val = math.exp(val - max_val)
                exp_values.append(exp_val)
                exp_sum += exp_val
            
            # 计算softmax概率
            for i in range(x.shape[self.dim]):
                temp_idx = list(idx)
                temp_idx[self.dim] = i
                result[tuple(temp_idx)].value = exp_values[i] / exp_sum
        
        return result

class ReLU(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor):
        return x.relu()

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor):
        return x.sigmoid()

class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.modules = modules
        for module in self.modules:
            self.register_module(module.__class__.__name__, module)
    def forward(self, x: Tensor):
        for module in self.modules:
            x = module(x)
        return x

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor):
        return x.leaky_relu(self.negative_slope)
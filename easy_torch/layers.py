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

class ParameterList(Module):
    def __init__(self, *params: Tensor):
        super().__init__()
        self.params = params
        for param in self.params:
            self.register_parameter(param.__class__.__name__, param)
    
    def forward(self, x: Tensor):
        for param in self.params:
            x = param(x)
        return x
    
    def __iter__(self):
        """使ParameterList可迭代"""
        return iter(self.params)
    
    def __len__(self):
        """返回参数数量"""
        return len(self.params)
    
    def __getitem__(self, idx):
        """支持索引访问"""
        return self.params[idx]

class Conv2d(Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        #ksize=3 ,stride=1,padding=0
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernels=ParameterList(*[Tensor.randn((in_channels,3,3),requires_grad=True) for _ in range(out_channels)])
    def forward(self, x: Tensor):
        """
        2D卷积层前向传播
        Args:
            x: 输入张量，形状为( in_channels,height, width, )
        Returns:
            输出张量，形状为(out_channels, height-2, width-2, )
        """
        rows=x.shape[1]
        cols=x.shape[2]
        out_height=rows-2
        out_width=cols-2
        
        # 创建正确顺序的输出数据
        out_data=[0] * (self.out_channels * out_height * out_width)
        
        # 对每个输出通道进行计算
        for c in range(self.out_channels):
            kernel=self.kernels[c]
            # 对输出张量的每个位置进行计算
            for row in range(out_height):
                for col in range(out_width):
                    # 计算输出数据中的索引
                    out_idx=c * out_height * out_width + row * out_width + col
                    
                    # 计算卷积
                    s=0
                    for i in range(self.in_channels):
                        for j in range(3):
                            for k in range(3):
                                s+=x[i,row+j,col+k]*kernel[i,j,k]
                    
                    # 将结果存储到正确的位置
                    out_data[out_idx]=s

        return Tensor((self.out_channels,out_height,out_width),requires_grad=True,data=out_data)

class MaxPool2d(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        """
        2D最大池化层前向传播
        Args:
            x: 输入张量，形状为(channels, height, width)
        Returns:
            输出张量，形状为(channels, height/2, width/2)
        """
        channels, in_height, in_width = x.shape
        
        # 计算输出尺寸
        out_height = in_height // 2
        out_width = in_width // 2
        
        # 创建输出数据
        out_data = [0] * (channels * out_height * out_width)
        
        # 对每个通道进行池化
        for c in range(channels):
            # 对输出张量的每个位置进行计算
            for out_h in range(out_height):
                for out_w in range(out_width):
                    # 计算输入张量中对应的区域
                    in_h_start = out_h * 2
                    in_w_start = out_w * 2
                    
                    # 找到2x2区域中的最大值
                    max_val = x[c, in_h_start, in_w_start]
                    
                    # 检查2x2区域内的所有值
                    for dh in range(2):
                        for dw in range(2):
                            in_h = in_h_start + dh
                            in_w = in_w_start + dw
                            # 确保不越界
                            if in_h < in_height and in_w < in_width:
                                val = x[c, in_h, in_w]
                                if val > max_val:
                                    max_val = val
                    
                    # 计算输出数据中的索引
                    out_idx = c * out_height * out_width + out_h * out_width + out_w
                    
                    # 将最大值存储到输出中
                    out_data[out_idx] = max_val
        
        # 创建并返回输出张量
        return Tensor((channels, out_height, out_width), requires_grad=x.requires_grad, data=out_data)
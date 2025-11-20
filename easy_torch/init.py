from .tensor import Tensor
import math
import random

def xavier_uniform_(tensor: Tensor):
    """使用Xavier均匀分布初始化张量
    
    Xavier初始化（也称为Glorot初始化）使用均匀分布U(-a, a)，
    其中a = sqrt(6 / (fan_in + fan_out))，fan_in是输入神经元数量，fan_out是输出神经元数量。
    
    Args:
        tensor: 要初始化的张量
    
    Returns:
        tensor: 初始化后的张量（原地修改）
    """
    # 计算fan_in和fan_out
    if len(tensor.shape) < 2:
        raise ValueError("Xavier initialization requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # 对于卷积核等多维张量，考虑所有输出维度
    for dim in tensor.shape[2:]:
        fan_in *= dim
        fan_out *= dim
    
    # 计算边界值
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    
    # 原地修改张量值
    for i in range(len(tensor.data)):
        tensor.data[i].value = random.uniform(-limit, limit)
    
    return tensor

def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    """使用Kaiming正态分布初始化张量
    
    Kaiming初始化（也称为He初始化）使用正态分布N(0, std)，
    其中std = sqrt(2 / (fan_in * (1 + a^2)))，fan_in是输入神经元数量，a是非线性激活函数的负斜率参数。
    
    Args:
        tensor: 要初始化的张量
        a: 激活函数的负斜率参数（默认为0.0，适用于ReLU）
        mode: 计算fan的模式，'fan_in'或'fan_out'（默认为'fan_in'）
        nonlinearity: 非线性激活函数类型（默认为'leaky_relu'）
    
    Returns:
        tensor: 初始化后的张量（原地修改）
    """
    # 计算fan_in和fan_out
    if len(tensor.shape) < 2:
        raise ValueError("Kaiming initialization requires at least 2D tensor")
    
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    
    # 对于卷积核等多维张量，考虑所有输出维度
    for dim in tensor.shape[2:]:
        fan_in *= dim
        fan_out *= dim
    
    # 根据mode选择fan
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'fan_in' or 'fan_out'")
    
    # 根据非线性激活函数调整增益
    if nonlinearity == 'relu' or nonlinearity == 'leaky_relu':
        # ReLU或Leaky ReLU的增益
        gain = math.sqrt(2.0 / (1 + a**2))
    elif nonlinearity == 'tanh':
        # tanh的增益
        gain = 5.0 / 3.0
    elif nonlinearity == 'sigmoid':
        # sigmoid的增益
        gain = 1.0
    else:
        gain = 1.0
    
    # 计算标准差
    std = gain / math.sqrt(fan)
    
    # 原地修改张量值（使用Box-Muller变换生成标准正态分布）
    for i in range(len(tensor.data)):
        # Box-Muller变换
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        tensor.data[i].value = std * z
    
    return tensor
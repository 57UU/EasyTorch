import typing
from .tensor import Tensor
from .grad_number import GradNumber

class Optimizer:
    def step(self):
        raise NotImplementedError()
    def zero_grad(self):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, params:typing.List[GradNumber], lr=0.01, momentum=0.0):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}  # 用于存储动量
        
        # 初始化速度
        for i, param in enumerate(self.params):
            self.velocities[i] = 0.0
            
    def step(self):
        for i, param in enumerate(self.params):
            # 更新速度
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad
            # 更新参数
            param.value -= self.velocities[i]
            
    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

class Adam(Optimizer):
    def __init__(self, params: typing.List[GradNumber], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        """
        Adam优化器实现
        
        参数:
            params: 需要优化的参数列表
            lr: 学习率 (默认: 0.001)
            betas: 用于计算梯度及其平方的移动平均的系数 (默认: (0.9, 0.999))
            eps: 数值稳定性的小常数 (默认: 1e-8)
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        # 初始化状态
        self.m = {}  # 梯度的移动平均
        self.v = {}  # 梯度平方的移动平均
        self.t = 0   # 时间步
        
        for i, param in enumerate(self.params):
            self.m[i] = 0.0
            self.v[i] = 0.0
    
    def step(self):
        """执行一步优化"""
        self.t += 1
        
        for i, param in enumerate(self.params):
            # 更新偏置修正的一阶矩估计
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            
            # 更新偏置修正的二阶矩估计
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
            
            # 计算偏置修正后的估计
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            # 更新参数
            param.value -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
    
    def zero_grad(self):
        """将所有参数的梯度清零"""
        for param in self.params:
            param.grad = 0.0
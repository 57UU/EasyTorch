from .tensor import Tensor
from .grad_number import GradNumber
import math

def cross_entropy_loss(pred: Tensor, target: Tensor) -> GradNumber:
    """
    pred : onehot batch x n (predictions after softmax)
    target : label batch (integer class indices)
    """
    batch_size = pred.shape[0]
    loss = GradNumber(0.0, requires_grad=True)
    
    for i in range(batch_size):
        target_idx = int(target[i].value)

        pred_prob = pred[i, target_idx]

        epsilon = 1e-15

        if pred_prob.value < epsilon:
            pred_prob.value = epsilon
        elif pred_prob.value > 1.0 - epsilon:
            pred_prob.value = 1.0 - epsilon
        else:
            pred_prob.value = pred_prob.value
            
        # 计算负对数似然: -log(p)
        sample_loss = GradNumber.log(pred_prob)
        sample_loss = GradNumber(0.0, requires_grad=True) - sample_loss  # 取负值
        
        # 累加损失
        loss = loss + sample_loss
    
    # 计算平均损失
    loss = loss / batch_size
    
    return loss
    
def softmax(x: Tensor) -> Tensor:
    """计算softmax函数"""
    # 创建结果张量
    result = Tensor(x.shape, x.requires_grad)
    
    # 对每个样本计算softmax
    for i in range(x.shape[0]):
        # 找到最大值以提高数值稳定性
        max_val = max(x[i, j].value for j in range(x.shape[1]))
        
        # 计算指数和
        exp_sum = 0.0
        exp_values = []
        for j in range(x.shape[1]):
            exp_val = math.exp(x[i, j].value - max_val)
            exp_values.append(exp_val)
            exp_sum += exp_val
        
        # 计算softmax概率
        for j in range(x.shape[1]):
            result[i, j].value = exp_values[j] / exp_sum
    
    return result

def mean_squared_error(pred: Tensor, target: Tensor) -> GradNumber:
    """
    pred : batch x n (predictions)
    target : batch x n (target values)
    """
    batch_size = pred.shape[0]
    loss = GradNumber(0.0, requires_grad=True)
    
    for i in range(batch_size):
        for j in range(pred.shape[1]):
            diff = pred[i, j] - target[i, j]
            loss = loss + diff * diff
    
    # 计算平均损失
    cases=batch_size*pred.shape[1]
    loss = loss / cases
    
    return loss
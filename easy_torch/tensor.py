from typing import Tuple, List, Optional, Union, Any
import math
import random
from .grad_number import GradNumber

class Tensor:
    def __init__(self, shape: Tuple[int, ...], requires_grad: bool = False,data=None):
        self.shape = shape
        self.requires_grad = requires_grad
        self.ndim = len(shape)
        if data is None:
            self.data = [GradNumber(0.0, requires_grad=requires_grad) for _ in range(math.prod(shape))]
        else:
            #verify
            if len(data) != math.prod(shape):
                raise ValueError(f"Expected {math.prod(shape)} elements, got {len(data)}")
            self.data = data
    
    def __str__(self) -> str:
        return f"Tensor(shape={self.shape})"
    
    def _get_index(self, indices: Tuple[int, ...]) -> int:
        """将多维索引转换为一维索引"""
        if len(indices) != self.ndim:
            raise ValueError(f"Expected {self.ndim} indices, got {len(indices)}")
        
        index = 0
        stride = 1
        for i in range(self.ndim - 1, -1, -1):
            if indices[i] < 0 or indices[i] >= self.shape[i]:
                raise IndexError(f"Index {indices[i]} out of bounds for dimension {i} with size {self.shape[i]}")
            index += indices[i] * stride
            stride *= self.shape[i]
        
        return index
    def relu(self):
        relu_value=[]
        for i in self.data:
            if i.value > 0:
                relu_value.append(i)
            else:
                relu_value.append(GradNumber.ZERO)
        return Tensor(self.shape,requires_grad=self.requires_grad,data=relu_value)
    def sigmoid(self):
        sigmoid_value=[]
        for i in self.data:
            sigmoid_value.append(i.sigmoid())
        return Tensor(self.shape,requires_grad=self.requires_grad,data=sigmoid_value)
    def leaky_relu(self, negative_slope: float = 0.01):
        leaky_relu_value=[]
        for i in self.data:
            if i.value < 0:
                leaky_relu_value.append(i * negative_slope)
            else:
                leaky_relu_value.append(i)
        return Tensor(self.shape,requires_grad=self.requires_grad,data=leaky_relu_value)

    def _get_indices(self) -> List[Tuple[int, ...]]:
        """获取所有可能的多维索引"""
        indices = []
        def _generate_indices(current_idx, dim):
            if dim == self.ndim:
                indices.append(tuple(current_idx))
                return
            
            for i in range(self.shape[dim]):
                _generate_indices(current_idx + [i], dim + 1)
        
        _generate_indices([], 0)
        return indices
    def item(self) -> Any:
        """返回张量的标量值"""
        if self.shape != (1,):
            raise ValueError("Only scalar tensors can be converted to Python scalars")
        return self.data[0]
    def __getitem__(self, indices: Union[int, Tuple[int, ...], slice]) -> Union['Tensor', GradNumber]:
        """获取张量中的元素或子张量"""
        # 处理单个整数索引
        if isinstance(indices, int):
            indices = (indices,)
        
        # 处理元组索引
        if isinstance(indices, tuple):
            # 如果索引数量等于维度数，返回单个元素
            if len(indices) == self.ndim:
                return self.data[self._get_index(indices)]
            
            # 否则返回子张量
            new_shape = []
            new_data = []
            
            # 计算新形状和数据
            for idx in self._get_indices():
                match = True
                for i, ind in enumerate(indices):
                    if isinstance(ind, int) and idx[i] != ind:
                        match = False
                        break
                    elif isinstance(ind, slice):
                        start, stop, step = ind.indices(self.shape[i])
                        if idx[i] not in range(start, stop, step):
                            match = False
                            break
                
                if match:
                    # 计算在新张量中的索引
                    new_idx = []
                    for i, ind in enumerate(indices):
                        if isinstance(ind, slice):
                            start, stop, step = ind.indices(self.shape[i])
                            new_idx.append((idx[i] - start) // step)
                        elif isinstance(ind, int):
                            continue  # 跳过固定维度
                    
                    if not new_idx:  # 如果所有维度都是固定索引，返回单个元素
                        return self.data[self._get_index(idx)]
                    
                    # 确保新形状正确
                    while len(new_shape) < len(new_idx):
                        new_shape.append(0)
                    
                    # 更新新形状
                    for i, ind in enumerate(indices):
                        if isinstance(ind, slice):
                            start, stop, step = ind.indices(self.shape[i])
                            size = len(range(start, stop, step))
                            if len(new_shape) <= i:
                                new_shape.append(size)
                            else:
                                new_shape[i] = size
                    
                    new_data.append(self.data[self._get_index(idx)])
            
            if not new_data:  # 如果没有匹配的元素
                return Tensor((0,), self.requires_grad)
            
            return Tensor(tuple(new_shape), self.requires_grad)._replace_data(new_data)
        
        # 处理切片索引
        elif isinstance(indices, slice):
            return self[(indices,)]
        
        else:
            raise TypeError(f"Invalid index type: {type(indices)}")
    
    def _replace_data(self, data: List[GradNumber]) -> 'Tensor':
        """替换张量的数据"""
        self.data = data
        return self
    
    def __setitem__(self, indices: Union[int, Tuple[int, ...]], value: Union[float, GradNumber]):
        """设置张量中的元素"""
        # 处理单个整数索引
        if isinstance(indices, int):
            indices = (indices,)
        
        # 处理元组索引
        if isinstance(indices, tuple):
            # 如果索引数量等于维度数，设置单个元素
            if len(indices) == self.ndim:
                if isinstance(value, GradNumber):
                    self.data[self._get_index(indices)] = value
                else:
                    self.data[self._get_index(indices)].value = value
                return
            
            # 否则设置子张量
            for idx in self._get_indices():
                match = True
                for i, ind in enumerate(indices):
                    if isinstance(ind, int) and idx[i] != ind:
                        match = False
                        break
                    elif isinstance(ind, slice):
                        start, stop, step = ind.indices(self.shape[i])
                        if idx[i] not in range(start, stop, step):
                            match = False
                            break
                
                if match:
                    if isinstance(value, GradNumber):
                        self.data[self._get_index(idx)] = value
                    else:
                        self.data[self._get_index(idx)].value = value
        
        # 处理切片索引
        elif isinstance(indices, slice):
            self[(indices,)] = value
        
        else:
            raise TypeError(f"Invalid index type: {type(indices)}")
    
    @staticmethod
    def rand(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """生成均匀分布的随机张量，值在[0, 1)区间"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = random.random()
        return result
    
    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """生成标准正态分布的随机张量（均值为0，标准差为1）"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            # 使用Box-Muller变换生成标准正态分布随机数
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            result.data[i].value = z0
        return result
    
    @staticmethod
    def uniform(shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, requires_grad: bool = False) -> 'Tensor':
        """生成指定区间的均匀分布随机张量"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = random.uniform(low, high)
        return result
    
    @staticmethod
    def normal(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, requires_grad: bool = False) -> 'Tensor':
        """生成指定均值和标准差的正态分布随机张量"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            # 使用Box-Muller变换生成标准正态分布随机数，然后缩放和平移
            u1 = random.random()
            u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            result.data[i].value = mean + std * z0
        return result
    
    @staticmethod
    def randint(shape: Tuple[int, ...], low: int, high: int, requires_grad: bool = False) -> 'Tensor':
        """生成指定区间的整数随机张量，范围[low, high)"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = random.randint(low, high - 1)  # randint包含high-1
        return result
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """生成全零张量"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = 0.0
        return result
    
    def zeros_like(self) -> 'Tensor':
        """生成与当前张量形状相同的全零张量"""
        return Tensor.zeros(self.shape, self.requires_grad)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """生成全一张量"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = 1.0
        return result
    
    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: float, requires_grad: bool = False) -> 'Tensor':
        """生成填充指定值的张量"""
        result = Tensor(shape, requires_grad)
        for i in range(len(result.data)):
            result.data[i].value = fill_value
        return result
    
    @staticmethod
    def eye(n: int, m: Optional[int] = None, requires_grad: bool = False) -> 'Tensor':
        """生成单位矩阵张量"""
        if m is None:
            m = n
        
        result = Tensor((n, m), requires_grad)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                if i == j:
                    result.data[idx].value = 1.0
                else:
                    result.data[idx].value = 0.0
        return result
    
    @staticmethod
    def arange(start: Union[int, float], stop: Optional[Union[int, float]] = None, step: Union[int, float] = 1, requires_grad: bool = False) -> 'Tensor':
        """生成等差数列张量"""
        if stop is None:
            stop = start
            start = 0
        
        # 计算长度
        if step == 0:
            raise ValueError("Step cannot be zero")
        
        length = max(0, math.ceil((stop - start) / step))
        result = Tensor((length,), requires_grad)
        
        for i in range(length):
            result.data[i].value = start + i * step
        
        return result
    
    @staticmethod
    def linspace(start: float, stop: float, num: int, requires_grad: bool = False) -> 'Tensor':
        """生成线性间隔的张量"""
        if num < 0:
            raise ValueError("Number of samples must be non-negative")
        
        if num == 0:
            return Tensor((0,), requires_grad)
        
        if num == 1:
            result = Tensor((1,), requires_grad)
            result.data[0].value = start
            return result
        
        step = (stop - start) / (num - 1)
        result = Tensor((num,), requires_grad)
        
        for i in range(num):
            result.data[i].value = start + i * step
        
        return result
    def apply(self, func) -> 'Tensor':
        """对张量的每个元素应用函数"""
        result = Tensor(self.shape, False)
        for i in range(len(self.data)):
            result.data[i].value = func(self.data[i].value)
        return result
    @staticmethod
    def from_list(data: List[float]) -> 'Tensor':
        """从列表创建张量"""
        t= Tensor((len(data),), requires_grad=False)
        for i in range(len(data)):
            t.data[i].value=data[i]
        return t
    @staticmethod
    def _broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
        """计算两个形状广播后的结果形状"""
        # 从右向左比较每个维度
        result_shape = []
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        
        for i in range(1, max_len + 1):
            dim1 = shape1[-i] if i <= len1 else 1
            dim2 = shape2[-i] if i <= len2 else 1
            
            if dim1 == dim2 or dim1 == 1 or dim2 == 1:
                result_shape.append(max(dim1, dim2))
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} are not broadcastable")
        
        return tuple(reversed(result_shape))
    
    def _broadcast_to(self, target_shape: Tuple[int, ...]) -> 'Tensor':
        """将当前张量广播到目标形状"""
        if self.shape == target_shape:
            return self
        
        # 检查是否可以广播
        try:
            broadcast_shape = Tensor._broadcast_shapes(self.shape, target_shape)
            if broadcast_shape != target_shape:
                raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")
        except ValueError:
            raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")
        
        # 创建新张量
        result = Tensor(target_shape, self.requires_grad)
        
        # 计算每个维度的广播步长
        strides = [1] * len(target_shape)
        for i in range(len(target_shape) - 1, -1, -1):
            if i < len(target_shape) - 1:
                strides[i] = strides[i + 1] * target_shape[i + 1]
        
        self_strides = [1] * len(self.shape)
        for i in range(len(self.shape) - 1, -1, -1):
            if i < len(self.shape) - 1:
                self_strides[i] = self_strides[i + 1] * self.shape[i + 1]
        
        # 填充数据
        for target_idx in result._get_indices():
            # 计算在源张量中的索引
            source_idx = []
            for i in range(len(target_shape)):
                target_dim = target_shape[i]
                if i >= len(self.shape):
                    # 源张量没有这个维度，使用0
                    source_idx.append(0)
                elif self.shape[i] == 1:
                    # 源张量在这个维度上大小为1，使用0
                    source_idx.append(0)
                else:
                    # 源张量在这个维度上大小与目标相同，使用相同的索引
                    source_idx.append(target_idx[i])
            
            # 设置值
            target_flat_idx = result._get_index(target_idx)
            source_flat_idx = self._get_index(tuple(source_idx)) if len(source_idx) == len(self.shape) else 0
            result.data[target_flat_idx] = self.data[source_flat_idx]
        
        return result
    
    @staticmethod
    def _broadcast_tensors(tensor1: 'Tensor', tensor2: 'Tensor') -> Tuple['Tensor', 'Tensor']:
        """将两个张量广播到兼容的形状"""
        target_shape = Tensor._broadcast_shapes(tensor1.shape, tensor2.shape)
        return tensor1._broadcast_to(target_shape), tensor2._broadcast_to(target_shape)
    
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """张量加法，支持广播"""
        # 处理标量加法
        if isinstance(other, (int, float)):
            other = Tensor((1,), self.requires_grad)
            other.data[0].value = other
        
        # 检查类型
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for +: 'Tensor' and '{type(other)}'")
        
        # 广播张量
        tensor1, tensor2 = Tensor._broadcast_tensors(self, other)
        
        # 创建结果张量
        result = Tensor(tensor1.shape, tensor1.requires_grad or tensor2.requires_grad)
        
        # 执行元素级加法
        for i in range(len(result.data)):
            result.data[i] = tensor1.data[i] + tensor2.data[i]
        
        return result
    
    def __radd__(self, other: Union[float, int]) -> 'Tensor':
        """反向加法，支持标量 + 张量"""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """张量减法，支持广播"""
        # 处理标量减法
        if isinstance(other, (int, float)):
            other = Tensor((1,), self.requires_grad)
            other.data[0].value = other
        
        # 检查类型
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for -: 'Tensor' and '{type(other)}'")
        
        # 广播张量
        tensor1, tensor2 = Tensor._broadcast_tensors(self, other)
        
        # 创建结果张量
        result = Tensor(tensor1.shape, tensor1.requires_grad or tensor2.requires_grad)
        
        # 执行元素级减法
        for i in range(len(result.data)):
            result.data[i] = tensor1.data[i] - tensor2.data[i]
        
        return result
    
    def __rsub__(self, other: Union[float, int]) -> 'Tensor':
        """反向减法，支持标量 - 张量"""
        # 处理标量减法
        if isinstance(other, (int, float)):
            other_tensor = Tensor((1,), self.requires_grad)
            other_tensor.data[0].value = other
            return other_tensor.__sub__(self)
        
        raise TypeError(f"Unsupported operand type(s) for -: '{type(other)}' and 'Tensor'")
    
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """张量乘法，支持广播"""
        # 处理标量乘法
        if isinstance(other, (int, float)):
            other = Tensor.from_list([other])
        
        # 检查类型
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for *: 'Tensor' and '{type(other)}'")
        
        # 广播张量
        tensor1, tensor2 = Tensor._broadcast_tensors(self, other)
        
        # 创建结果张量
        result = Tensor(tensor1.shape, tensor1.requires_grad or tensor2.requires_grad)
        
        # 执行元素级乘法
        for i in range(len(result.data)):
            result.data[i] = tensor1.data[i] * tensor2.data[i]
        
        return result
    
    def __rmul__(self, other: Union[float, int]) -> 'Tensor':
        """反向乘法，支持标量 * 张量"""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """张量除法，支持广播"""
        # 处理标量除法
        if isinstance(other, (int, float)):
            other = Tensor((1,), self.requires_grad)
            other.data[0].value = other
        
        # 检查类型
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for /: 'Tensor' and '{type(other)}'")
        
        # 广播张量
        tensor1, tensor2 = Tensor._broadcast_tensors(self, other)
        
        # 创建结果张量
        result = Tensor(tensor1.shape, tensor1.requires_grad or tensor2.requires_grad)
        
        # 执行元素级除法
        for i in range(len(result.data)):
            result.data[i] = tensor1.data[i] / tensor2.data[i]
        
        return result
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Tensor':
        """反向除法，支持标量 / 张量"""
        # 处理标量除法
        if isinstance(other, (int, float)):
            other_tensor = Tensor((1,), self.requires_grad)
            other_tensor.data[0].value = other
            return other_tensor.__truediv__(self)
        
        raise TypeError(f"Unsupported operand type(s) for /: '{type(other)}' and 'Tensor'")
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法，支持广播"""
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported operand type(s) for matmul: 'Tensor' and '{type(other)}'")
        
        # 检查维度
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("Both tensors must have at least 1 dimension for matmul")
        
        # 处理标量情况
        if self.ndim == 1 and other.ndim == 1:
            # 点积
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} are not aligned for dot product")
            
            result = Tensor((), self.requires_grad or other.requires_grad)
            result.data[0] = GradNumber(0.0, result.requires_grad)
            
            for i in range(self.shape[0]):
                result.data[0] = result.data[0] + (self.data[i] * other.data[i])
            
            return result
        
        # 处理矩阵向量乘法
        elif self.ndim == 2 and other.ndim == 1:
            # 矩阵 @ 向量
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} are not aligned for matrix-vector multiplication")
            
            result = Tensor((self.shape[0],), self.requires_grad or other.requires_grad)
            
            for i in range(self.shape[0]):
                result.data[i] = GradNumber(0.0, result.requires_grad)
                for j in range(self.shape[1]):
                    result.data[i] = result.data[i] + (self.data[i * self.shape[1] + j] * other.data[j])
            
            return result
        
        # 处理向量矩阵乘法
        elif self.ndim == 1 and other.ndim == 2:
            # 向量 @ 矩阵
            if self.shape[0] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} are not aligned for vector-matrix multiplication")
            
            result = Tensor((other.shape[1],), self.requires_grad or other.requires_grad)
            
            for j in range(other.shape[1]):
                result.data[j] = GradNumber(0.0, result.requires_grad)
                for i in range(other.shape[0]):
                    result.data[j] = result.data[j] + (self.data[i] * other.data[i * other.shape[1] + j])
            
            return result
        
        # 处理矩阵矩阵乘法
        elif self.ndim == 2 and other.ndim == 2:
            # 矩阵 @ 矩阵
            if self.shape[1] != other.shape[0]:
                raise ValueError(f"Shapes {self.shape} and {other.shape} are not aligned for matrix multiplication")
            
            result = Tensor((self.shape[0], other.shape[1]), self.requires_grad or other.requires_grad)
            
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    result.data[i * other.shape[1] + j] = GradNumber(0.0, result.requires_grad)
                    for k in range(self.shape[1]):
                        result.data[i * other.shape[1] + j] = result.data[i * other.shape[1] + j] + (
                            self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j]
                        )
            
            return result
        
        # 处理批量矩阵乘法
        else:
            # 批量矩阵乘法
            # 广播除了最后两个维度外的所有维度
            batch_shape1 = self.shape[:-2] if self.ndim > 2 else ()
            batch_shape2 = other.shape[:-2] if other.ndim > 2 else ()
            
            try:
                batch_shape = Tensor._broadcast_shapes(batch_shape1, batch_shape2)
            except ValueError:
                raise ValueError(f"Batch shapes {batch_shape1} and {batch_shape2} are not broadcastable")
            
            # 检查矩阵维度是否兼容
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(f"Matrix dimensions {self.shape[-1]} and {other.shape[-2]} are not aligned")
            
            # 创建结果张量
            result_shape = batch_shape + (self.shape[-2], other.shape[-1])
            result = Tensor(result_shape, self.requires_grad or other.requires_grad)
            
            # 计算批量矩阵乘法
            for batch_idx in result._get_indices()[:len(batch_shape)]:
                # 计算批量索引的扁平化索引
                batch_flat_idx = 0
                batch_stride = 1
                for i in range(len(batch_shape) - 1, -1, -1):
                    batch_flat_idx += batch_idx[i] * batch_stride
                    batch_stride *= batch_shape[i]
                
                # 计算在源张量中的批量索引
                self_batch_idx = []
                other_batch_idx = []
                for i in range(len(batch_shape)):
                    if i >= len(batch_shape1):
                        self_batch_idx.append(0)
                    elif batch_shape1[i] == 1:
                        self_batch_idx.append(0)
                    else:
                        self_batch_idx.append(batch_idx[i])
                    
                    if i >= len(batch_shape2):
                        other_batch_idx.append(0)
                    elif batch_shape2[i] == 1:
                        other_batch_idx.append(0)
                    else:
                        other_batch_idx.append(batch_idx[i])
                
                # 计算在源张量中的扁平化批量索引
                self_batch_flat_idx = 0
                self_batch_stride = 1
                for i in range(len(self_batch_idx) - 1, -1, -1):
                    self_batch_flat_idx += self_batch_idx[i] * self_batch_stride
                    self_batch_stride *= batch_shape1[i] if i < len(batch_shape1) else 1
                
                other_batch_flat_idx = 0
                other_batch_stride = 1
                for i in range(len(other_batch_idx) - 1, -1, -1):
                    other_batch_flat_idx += other_batch_idx[i] * other_batch_stride
                    other_batch_stride *= batch_shape2[i] if i < len(batch_shape2) else 1
                
                # 计算矩阵乘法
                for i in range(self.shape[-2]):
                    for j in range(other.shape[-1]):
                        result_idx = batch_idx + (i, j)
                        result_flat_idx = result._get_index(result_idx)
                        result.data[result_flat_idx] = GradNumber(0.0, result.requires_grad)
                        
                        for k in range(self.shape[-1]):
                            self_idx = self_batch_idx + (i, k)
                            other_idx = other_batch_idx + (k, j)
                            
                            self_flat_idx = self._get_index(self_idx)
                            other_flat_idx = other._get_index(other_idx)
                            
                            result.data[result_flat_idx] = result.data[result_flat_idx] + (
                                self.data[self_flat_idx] * other.data[other_flat_idx]
                            )
            
            return result
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """矩阵乘法运算符 @"""
        return self.matmul(other)

    def zero_grad(self):
        """清零梯度"""
        for i in range(len(self.data)):
            self.data[i].grad = 0
    
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """计算张量元素的和"""
        if axis is None:
            # 计算所有元素的和
            result = Tensor((1,), self.requires_grad)
            result.data[0] = GradNumber(0.0, result.requires_grad)
            
            for i in range(len(self.data)):
                result.data[0] = result.data[0] + self.data[i]
            
            return result
        else:
            # 沿指定轴计算和
            if isinstance(axis, int):
                axis = (axis,)
            
            # 确保轴是有效的
            for ax in axis:
                if ax < 0 or ax >= self.ndim:
                    raise ValueError(f"Axis {ax} is out of bounds for tensor of dimension {self.ndim}")
            
            # 计算结果形状
            result_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    result_shape[ax] = 1
                else:
                    result_shape.pop(ax)
            
            result_shape = tuple(result_shape)
            result = Tensor(result_shape, self.requires_grad)
            
            # 初始化结果张量为0
            for i in range(len(result.data)):
                result.data[i] = GradNumber(0.0, result.requires_grad)
            
            # 计算和
            for idx in self._get_indices():
                # 计算在结果张量中的索引
                result_idx = list(idx)
                for ax in sorted(axis, reverse=True):
                    if not keepdims:
                        result_idx.pop(ax)
                    else:
                        result_idx[ax] = 0
                
                result_idx = tuple(result_idx)
                result_flat_idx = result._get_index(result_idx) if result_idx else 0
                self_flat_idx = self._get_index(idx)
                
                result.data[result_flat_idx] = result.data[result_flat_idx] + self.data[self_flat_idx]
            
            return result
    
    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """计算张量元素的平均值"""
        if axis is None:
            # 计算所有元素的平均值
            total = self.sum()
            count = len(self.data)
            result = Tensor((), self.requires_grad)
            result.data[0] = total.data[0] / count
            return result
        else:
            # 沿指定轴计算平均值
            if isinstance(axis, int):
                axis = (axis,)
            
            # 计算指定轴上的元素数量
            count = 1
            for ax in axis:
                count *= self.shape[ax]
            
            # 计算和并除以数量
            total = self.sum(axis, keepdims)
            result = total / count
            return result
    
    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """计算张量元素的最大值"""
        if axis is None:
            # 计算所有元素的最大值
            if len(self.data) == 0:
                raise ValueError("Cannot compute max of empty tensor")
            
            result = Tensor((), self.requires_grad)
            max_val = self.data[0].value
            max_idx = 0
            
            for i in range(1, len(self.data)):
                if self.data[i].value > max_val:
                    max_val = self.data[i].value
                    max_idx = i
            
            result.data[0] = GradNumber(max_val, result.requires_grad)
            return result
        else:
            # 沿指定轴计算最大值
            if isinstance(axis, int):
                axis = (axis,)
            
            # 确保轴是有效的
            for ax in axis:
                if ax < 0 or ax >= self.ndim:
                    raise ValueError(f"Axis {ax} is out of bounds for tensor of dimension {self.ndim}")
            
            # 计算结果形状
            result_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    result_shape[ax] = 1
                else:
                    result_shape.pop(ax)
            
            result_shape = tuple(result_shape)
            result = Tensor(result_shape, self.requires_grad)
            
            # 计算最大值
            for result_idx in result._get_indices():
                # 扩展结果索引到源张量的形状
                source_idx_template = list(result_idx)
                for ax in sorted(axis):
                    if not keepdims:
                        source_idx_template.insert(ax, 0)
                    else:
                        source_idx_template[ax] = 0
                
                # 初始化最大值
                source_idx = tuple(source_idx_template)
                max_val = self.data[self._get_index(source_idx)].value
                
                # 在指定轴上查找最大值
                for ax in axis:
                    for i in range(1, self.shape[ax]):
                        source_idx_list = list(source_idx_template)
                        source_idx_list[ax] = i
                        source_idx = tuple(source_idx_list)
                        val = self.data[self._get_index(source_idx)].value
                        if val > max_val:
                            max_val = val
                
                # 设置结果
                result_flat_idx = result._get_index(result_idx) if result_idx else 0
                result.data[result_flat_idx] = GradNumber(max_val, result.requires_grad)
            
            return result
    
    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        """计算张量元素的最小值"""
        if axis is None:
            # 计算所有元素的最小值
            if len(self.data) == 0:
                raise ValueError("Cannot compute min of empty tensor")
            
            result = Tensor((), self.requires_grad)
            min_val = self.data[0].value
            min_idx = 0
            
            for i in range(1, len(self.data)):
                if self.data[i].value < min_val:
                    min_val = self.data[i].value
                    min_idx = i
            
            result.data[0] = GradNumber(min_val, result.requires_grad)
            return result
        else:
            # 沿指定轴计算最小值
            if isinstance(axis, int):
                axis = (axis,)
            
            # 确保轴是有效的
            for ax in axis:
                if ax < 0 or ax >= self.ndim:
                    raise ValueError(f"Axis {ax} is out of bounds for tensor of dimension {self.ndim}")
            
            # 计算结果形状
            result_shape = list(self.shape)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    result_shape[ax] = 1
                else:
                    result_shape.pop(ax)
            
            result_shape = tuple(result_shape)
            result = Tensor(result_shape, self.requires_grad)
            
            # 计算最小值
            for result_idx in result._get_indices():
                # 扩展结果索引到源张量的形状
                source_idx_template = list(result_idx)
                for ax in sorted(axis):
                    if not keepdims:
                        source_idx_template.insert(ax, 0)
                    else:
                        source_idx_template[ax] = 0
                
                # 初始化最小值
                source_idx = tuple(source_idx_template)
                min_val = self.data[self._get_index(source_idx)].value
                
                # 在指定轴上查找最小值
                for ax in axis:
                    for i in range(1, self.shape[ax]):
                        source_idx_list = list(source_idx_template)
                        source_idx_list[ax] = i
                        source_idx = tuple(source_idx_list)
                        val = self.data[self._get_index(source_idx)].value
                        if val < min_val:
                            min_val = val
                
                # 设置结果
                result_flat_idx = result._get_index(result_idx) if result_idx else 0
                result.data[result_flat_idx] = GradNumber(min_val, result.requires_grad)
            
            return result
    
    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        # 计算新形状的元素数量
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        # 检查元素数量是否匹配
        current_size = len(self.data)
        if new_size != current_size:
            raise ValueError(f"Cannot reshape tensor of size {current_size} into shape {new_shape}")
        
        # 创建新张量
        result = Tensor(new_shape, self.requires_grad)
        result.data = self.data.copy()
        
        return result
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """转置张量"""
        if axes is None:
            # 默认反转所有轴
            axes = tuple(range(self.ndim - 1, -1, -1))
        
        # 检查轴是否有效
        if len(axes) != self.ndim:
            raise ValueError(f"Number of axes {len(axes)} does not match tensor dimension {self.ndim}")
        
        for ax in axes:
            if ax < 0 or ax >= self.ndim:
                raise ValueError(f"Axis {ax} is out of bounds for tensor of dimension {self.ndim}")
        
        if len(set(axes)) != self.ndim:
            raise ValueError("Axes must be unique")
        
        # 计算新形状
        new_shape = tuple(self.shape[ax] for ax in axes)
        
        # 创建新张量
        result = Tensor(new_shape, self.requires_grad)
        
        # 填充数据
        for new_idx in result._get_indices():
            # 计算在原张量中的索引
            old_idx = [0] * self.ndim
            for i, ax in enumerate(axes):
                old_idx[ax] = new_idx[i]
            
            old_idx = tuple(old_idx)
            new_flat_idx = result._get_index(new_idx)
            old_flat_idx = self._get_index(old_idx)
            
            result.data[new_flat_idx] = self.data[old_flat_idx]
        
        return result
    
    def T(self) -> 'Tensor':
        """转置2D张量"""
        if self.ndim != 2:
            raise ValueError("T property is only defined for 2D tensors")
        
        return self.transpose((1, 0))
    
    def log(self) -> 'Tensor':
        """计算张量元素的自然对数"""
        result = Tensor(self.shape, self.requires_grad)
        for i in range(len(self.data)):
            result.data[i] = GradNumber.log(self.data[i])
        return result
    
    def exp(self) -> 'Tensor':
        """计算张量元素的指数"""
        result = Tensor(self.shape, self.requires_grad)
        for i in range(len(self.data)):
            result.data[i] = GradNumber.exp(self.data[i])
        return result
    @staticmethod
    def stack(tensors:List['Tensor'])->'Tensor':
        l=[]
        for t in tensors:
            l.extend(t.data)
        newShape=(len(tensors),)+tensors[0].shape
        result=Tensor(newShape,data=l)
        return result

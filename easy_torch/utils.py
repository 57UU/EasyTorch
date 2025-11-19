from .tensor import Tensor

class Dataset:
    def __len__(self):
        raise NotImplementedError()
    def __getitem__(self, idx):
        raise NotImplementedError()

import random

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset: Dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.indices = list(range(self.length))
        if self.shuffle:
            random.shuffle(self.indices)
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.index >= self.length:
            # 重置索引，为下一轮迭代准备
            self.index = 0
            # 如果启用了shuffle，在下一轮迭代前重新打乱索引
            if self.shuffle:
                random.shuffle(self.indices)
            raise StopIteration()
        
        x = []
        y = []
        # 计算本轮要取的索引范围
        end_index = min(self.index + self.batch_size, self.length)
        
        # 使用预计算的索引列表访问数据
        for i in range(self.index, end_index):
            # 使用打乱后的索引访问数据集
            _x, _y = self.dataset[self.indices[i]]
            x.append(_x)
            y.append(_y)
        
        self.index = end_index
        batch_x = Tensor.stack(x)
        batch_y = Tensor.stack(y)
        return batch_x, batch_y

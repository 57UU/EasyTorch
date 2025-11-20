import math
import re
from easy_torch.layers import *
from easy_torch.optim import *
from easy_torch.utils import *
import easy_torch.function as F
import random
class LinearDataset(Dataset):
    def __init__(self):
        start=0
        end=10
        step=0.1
        
        self.x=[]
        self.y=[]
        while start<=end:
            self.x.append(Tensor.from_list([start]))
            self.y.append(5-Tensor.from_list([start+random.uniform(-0.5,0.5)]))
            start+=step
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
class SqureDataset(Dataset):
    def __init__(self):
        start=0
        end=10
        step=0.2
        
        self.x=[]
        self.y=[]
        while start<=end:
            self.x.append(Tensor.from_list([start]))
            self.y.append(Tensor.from_list([math.sin(start*0.5)]))
            start+=step
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
dataset=SqureDataset()
dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
class Model(Module):
    def __init__(self):
        super().__init__()
        self.layers=Sequential(
            Linear(1, 20),
            LeakyReLU(0.1),
            Linear(20, 20),
            LeakyReLU(0.1),
            Linear(20, 1),
        )
    def forward(self, x: Tensor):
        return self.layers(x)

model=Model()


optimizer=Adam(model.parameters(),lr=0.01)
epoches=100
for epoch in range(epoches):
    for x,y in dataloader:
        y_pred=model(x)
        loss=F.mean_squared_error(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.value}")


#draw
import matplotlib.pyplot as plt
plt.scatter([i[0].item().value for i in dataset],[i[1].item().value for i in dataset])
plt.plot([i[0].item().value for i in dataset],[model(Tensor.from_list([i[0].item().value])).item().value for i in dataset])
plt.savefig("model.png")

from easy_torch.layers import *
from easy_torch.optim import *
from easy_torch.utils import *
import easy_torch.function as F

import pickle
import numpy as np

dataset="dataset/data_batch_1"

class CIFAR_10(Dataset):
    def __init__(self,data_path,limit=100):
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images=dict[b"data"].reshape(-1,3,32,32).astype(np.float32)/255.0
            labels=dict[b"labels"]
            self.labels=[Tensor.from_cache(label) for label in labels]
            #convert to tensor
            self.images=[]
            print("loading")
            for i in range(images.shape[0]):
                if limit is not None and i>=limit:
                    break
                self.images.append(Tensor.from_numpy(images[i]))
            print("loading done")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
        return image,label



cifar_10=CIFAR_10(dataset)


class CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(3,6)
        self.conv2=Conv2d(6,12)
        self.pool=MaxPool2d()
        self.fc1=Linear(432,100)
        self.fc2=Linear(100,10) 
        self.softmax=Softmax()

    def forward(self, x: Tensor):
        # 3x32x32
        x=self.conv1(x) # 6x30x30
        x=self.pool(x) # 6x15x15
        x=x.relu()
        x=self.conv2(x) # 12x13x13
        x=self.pool(x) # 12x6x6
        x=x.relu()
        x=x.reshape((1,432)) # 432
        x=self.fc1(x) # 100
        x=x.relu()
        x=self.fc2(x) # 10
        x=self.softmax(x)
        return x

model=CNN()
optimizer=Adam(model.parameters(),lr=0.01)
epochs=10
mini_batch_size=32
for epoch in range(epochs):
    for i in range(len(cifar_10)):
        image,label=cifar_10[i]
        image=image
        pred=model(image)
        loss=F.cross_entropy_loss(pred,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch} loss {loss.value}")


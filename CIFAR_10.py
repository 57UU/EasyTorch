from easy_torch.layers import *
from easy_torch.optim import *
from easy_torch.utils import *
import easy_torch.function as F

import pickle

dataset="dataset/data_batch_1"

class CIFAR_10(Dataset):
    def __init__(self,data_path):
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images=dict[b"data"].reshape(-1,3,32,32).astype(np.float32)/255.0
            self.labels=dict[b"labels"]
            #convert to tensor
            self.images=Tensor(images)
            self.labels=Tensor(self.labels)

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
        return image,label



cifar_10=CIFAR_10(dataset)

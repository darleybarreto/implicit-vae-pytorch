import torch
from torch.utils.data import Dataset
from torch._utils import _accumulate

import numpy as np

class BatchSubset(Dataset):
    """
    :dataset (Dataset):              The whole Dataset
    :sub_indices (sequence):         The new order in wich the data must be 
    """
    def __init__(self, dataset: Dataset,  indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        return self.dataset[idx]

    def __len__(self):
        return len(self.indices)

class Binarize(object):
    def __init__(self, tresh=0.5):
        self.tresh = torch.tensor(tresh)

    def __call__(self,img):
        return (img > self.tresh).float()

def expand_indices(indices: np.ndarray, batch_size: int) -> np.ndarray:
    """
    :indices:       indices to be expanded based on batch_size
    :batch_size:    size of the batch
    
    This function expands the vector indices such that each i in indices
    becomes a vector of size batch_size.
    V[i] + j from range(batch_size), that is V[i] + 0, V[i] + 1, ... V[i] + len(batch_size) - 1.
    """

    full_matrix = np.array([indices + i for i in range(batch_size)]).T.flatten()
    return full_matrix

def random_batch_split(dataset, batch_size: int, lengths: list) -> list:
    """
    :dataset:       torch dataset
    :batch_size:    size of the batch
    :lengths:       lengths of batches

    This function shuffles the dataset based on the batches. We exchange the order in which
    they are, then for each head of a batch (batch_indices[i]), we compute all the batch_size
    number for each i. For exmaple, if batch_indices[i] = 320, than the batch i begins at position 320
    and goes to 320 + (batch_size - 1) in the array indices.
    """
    batch_indices = np.random.permutation(np.linspace(0, len(dataset)+1, endpoint=False,num=len(dataset)//batch_size,dtype=int)) # random number of first element of a batch
    return [expand_indices(batch_indices[offset - length:offset], batch_size) for offset, length in zip(_accumulate(lengths), lengths)]
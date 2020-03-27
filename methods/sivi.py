import torch

import torch.nn.functional as F

def sivi(data, model, J, K, warm_up):
    recons, logpz, loss = model(data,warm_up,J,K)

    loss.backward()

    return logpz, loss
import torch

import torch.nn.functional as F

def sivi(data, model, optimizer, J, K, warm_up):
    recons, logpz, loss = model(data,warm_up,J,K)

    loss.backward()
    
    optimizer.step()

    return logpz.mean().item(), loss
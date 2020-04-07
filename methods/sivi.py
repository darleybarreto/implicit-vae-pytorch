import torch

import torch.nn.functional as F

def sivi(data, model, optimizer, scheduler, split, J, K):
    if 'Training' == split:
        optimizer.zero_grad()

    recons, logpz, loss = model(data,1,J,K)

    if 'Training' == split:
        loss.backward()

        optimizer.step()

    return logpz.mean().item(), loss.item()
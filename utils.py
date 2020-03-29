import math
import torch

import numpy as np

import torch.nn.functional as F

class Binarize(object):
    def __init__(self, tresh=0.5):
        self.tresh = torch.tensor(tresh)

    def __call__(self,img):
        return (img > self.tresh).float()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        output = self.dataset[idx]
        if output.__class__ is tuple and len(output) > 1:
            output = output[0]
        
        return idx, output.view(-1)


def compute_llh_vae(T, S, model, test_data):
    # Compute an importance sampling estimate of the log-evidence on test
    #  T: number of samples for the log q(z) term
    #  S: number of samples for the importance sampling approximation
    model.eval()

    px = torch.zeros(len(test_data),1)
    z_dim = model.z_dim

    with torch.no_grad():
        for ns in range(len(test_data)):
            # Sample epsilon and pass through the NN

            repated_data = test_data[ns].repeat(T,*(test_data[ns].squeeze().shape))
            _, tr_epsilon_all = model(repated_data)
            
            # Sample z
            repeated_data2 = test_data[ns].repeat(S,*(test_data[ns].squeeze().shape))
            z, _ = model(repeated_data2)
            
            # Approximate log q(z)
            d1 = tr_epsilon_all/model.sigma
            d2 = z/model.sigma

            diffs2 = ((-2*d1*d2.transpose(1,0))+torch.sum(d1*d1,dim=1))+torch.sum(d2*d2,dim=1).transpose(1,0)

            logq = - 0.5*z_dim*math.log2(2*math.pi) - torch.sum(torch.log(model.sigma)) - 0.5*diffs2
            logq = torch.logsumexp(logq,dim=0).transpose(1,0) - math.log2(T)
            
            # Evaluate the log-joint
            logjoint = model.p_x_z_logdensity(repeated_data2, z)
            
            # Importance sampling term
            px[ns] = torch.logsumexp(logjoint - logq, dim=0) - math.log2(S)

    mean_px = torch.mean(px).item()

    return mean_px
import math

import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli

import torch.nn.functional as F

EPS = 1e-10

class SIVAE(nn.Module):
    def __init__(self, input_dim, device, z_dim=64, noise_dim = [150,100,50]):
        super(SIVAE, self).__init__()

        self.noise = Bernoulli(probs=0.5)
        self.z_dim = z_dim
        self.noise_dim = noise_dim

        self.hiddel_l3 = nn.Sequential( nn.Linear(input_dim + noise_dim[0],500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU(),
                                        nn.Linear(500,noise_dim[0]),
                                        nn.ReLU()
                                        )

        self.hiddel_l2 = nn.Sequential( nn.Linear(input_dim + noise_dim[0] + noise_dim[1],500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU(),
                                        nn.Linear(500,noise_dim[1]),
                                        nn.ReLU()
                                        )

        self.hiddel_l1 = nn.Sequential( nn.Linear(input_dim + noise_dim[1] + noise_dim[2],500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU()
                                        )

        self.mu = nn.Linear(500,z_dim)
        
        self.z_logvar = nn.Sequential( nn.Linear(input_dim,500),
                                    nn.ReLU(),
                                    nn.Linear(500,500),
                                    nn.ReLU(),
                                    nn.Linear(500,z_dim))

        self.decoder = nn.Sequential( nn.Linear(z_dim,500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU(),
                                        nn.Linear(500,500),
                                        nn.ReLU(),
                                        nn.Linear(500,input_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std

    def sample_psi(self, inputs, D):
        inputs_expanded = inputs.unsqueeze(dim=1).repeat(1,D,1)

        e3 = self.noise.sample((inputs.size(0)*D*self.noise_dim[0])).view(inputs.size(0),D,self.noise_dim[0]).float()
        h3 = self.hiddel_l3(torch.cat((e3,inputs_expanded),dim=2))

        e2 = self.noise.sample((inputs.size(0)*D*self.noise_dim[1])).view(inputs.size(0),D,self.noise_dim[1]).float()
        h2 = self.hiddel_l2(torch.cat((h3,e2,inputs_expanded),dim=2))

        e1 = self.noise.sample((inputs.size(0)*D*self.noise_dim[2])).view(inputs.size(0),D,self.noise_dim[2]).float()
        h1 = self.hiddel_l1(torch.cat((h2,e1,inputs_expanded),dim=2))

        psi_iw = self.mu(h1)

        return psi_iw

    def forward(self,inputs,warm_up,J,K):
        
        psi_iw = self.sample_psi(inputs,K)
        
        z_logv = self.z_logvar(inputs).unsqueeze(dim=1)
        z_logv_iw = z_logv.repeat(1,K,1)

        sigma_iw1 = torch.exp(z_logv_iw/2)
        sigma_iw2 = sigma_iw1.unsqueeze(dim=1).repeat(1,1,J+1,1)

        z_sample_iw = self.reparameterize(psi_iw,sigma_iw1).unsqueeze(dim=2).repeat(1,1,J+1,1)

        psi_iw_star_ = self.sample_psi(inputs,J).unsqueeze(dim=1).repeat(1,K,1,1)
        psi_iw_star = torch.cat((psi_iw_star_,psi_iw.unsqueeze(dim=2),2))

        ker = torch.exp(-0.5*torch.sum(torch.pow(z_sample_iw-psi_iw_star)/torch.pow(sigma_iw2+EPS),3))
        log_H_iw = torch.log(torch.mean(ker,dim=2))-0.5*torch.sum(z_logv_iw,2)

        log_prior_iw = -0.5*torch.sum(torch.pow(z_sample_iw),2)

        x_iw = inputs.unsqueeze(dim=1).repeat(1,K,1)

        logits_x_iw = self.decoder(z_sample_iw)
        p_x_iw = Bernoulli(logits=logits_x_iw) 

        reconstruct_iw = p_x_iw.mean()
        log_lik_iw = torch.sum( x_iw * torch.log(reconstruct_iw + EPS)
                    + (1-x_iw) * torch.log(1 - reconstruct_iw + EPS),2)


        loss_iw = -torch.logsumexp(log_lik_iw+(log_prior_iw-log_H_iw)*warm_up,1)+torch.log(torch.tensor(K).float())

        return reconstruct_iw, log_lik_iw, loss_iw.mean()
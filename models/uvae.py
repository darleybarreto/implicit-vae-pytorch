import math

import torch
from torch import nn
from torch.distributions.normal import Normal

import torch.nn.functional as F

class UVAE(nn.Module):
    def __init__(self, in_dim, device, z_dim=10,likelihood='Bernoulli'):
        super(UVAE, self).__init__()

        self.likelihood = likelihood
        self.hidden_dim_enc = 200
        self.hidden_dim_dec = 200
        self.z_dim = z_dim
        self.device = device
        # prior q(\epsilon) #
        self.q_eps = Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device))
        ##

        # auxiliary q(u) #
        self.q_u = Normal(torch.tensor(0.,device=device), torch.tensor(1.,device=device))
        ##

        # encoder q_\theta(z|\epsilon) #
        #   self.mu_eps == \mu_\theta(\epsilon)
        #   self.sigma == diag(\sigma)
        self.mu_eps = nn.Sequential(
            nn.Linear(in_dim + z_dim, self.hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_enc, self.hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_enc,z_dim)
        )

        # Learning the sqrt(\Sigma)
        self.sigma = nn.Parameter(0.5 * torch.ones((1, z_dim)))

        # decoder p_\phi(x|z) #
        self.p_phi = nn.Sequential(
            nn.Linear(z_dim, self.hidden_dim_dec),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_dec, self.hidden_dim_dec),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_dec, in_dim + z_dim),
            nn.Sigmoid()
        )
        ##

    def forward(self, x, epsilon=None):
        if epsilon is None:
            dim_to_cat = 1

            # u_s ~ q(u)
            u = self.q_u.sample((x.size(0), self.z_dim))
            
            # \epsilon_s ~ q(\epsilon)
            epsilon = self.q_eps.sample((x.size(0), self.z_dim))
        
        else:
            dim_to_cat = 2
            u = None

        # z_s = h_\theta(u_s; \epsilon_s)
        mu_eps = self.mu_eps(torch.cat([x,epsilon],dim=dim_to_cat))

        if not u is None:
            z = mu_eps + self.sigma*u
        else:
            z = None

        return z, mu_eps

    def q_z_e_logdensity(self,x,z,epsilon):
        k = epsilon.size(1)

        inputs = torch.cat([x,epsilon],dim=1)
        mu_eps = self.mu_eps(inputs)

        diff_pow = ((z - mu_eps)/self.sigma)**2
        
        q_z_e_logdensity = - 0.5*torch.sum( diff_pow, 1) - 0.5*torch.sum(epsilon**2, 1)

        gradepsilon = torch.autograd.grad(q_z_e_logdensity,inputs,grad_outputs=torch.ones(q_z_e_logdensity.shape).to(self.device),retain_graph=True)[0]

        return q_z_e_logdensity, gradepsilon[:,x.size(1):]

    def p_x_z_logdensity(self,x,z):
        x_hat = self.p_phi(z)

        if self.likelihood == 'Categorical': # % categorical output data          
            M = torch.max(x_hat, 1)
            p_x_z_logdensity = torch.sum( x*x_hat, 1) - M  - torch.log(torch.sum(torch.exp(x_hat - M), 1))

        elif self.likelihood == 'Bernoulli': # binary output data 
            p_x_z_logdensity = torch.sum(x_hat*torch.log(x_hat + 1e-10) + (x_hat)*torch.log(1 - x_hat + 1e-10), 1)

        # standard normal prior over the latent variables 
        p_x_z_logdensity -=  0.5*z.size(1)*math.log2(2*math.pi) - 0.5*torch.sum((z**2),1) 

        return p_x_z_logdensity
        
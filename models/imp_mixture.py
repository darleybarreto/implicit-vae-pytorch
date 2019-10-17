import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

class ImpMix(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim=10):
        super(ImpMix, self).__init__()

        self.hidden_dim_enc = 200

        # prior q(\epsilon) #
        self.q_eps = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        ##

        # auxiliary q(u) #
        self.q_u = MultivariateNormal(torch.zeros(z_dim), torch.eye(z_dim))
        ##

        # encoder q_\theta(z|\epsilon) #
        #   self.mu_eps == \mu_\theta(\epsilon)
        #   self.sigma == diag(\sigma)
        self.mu_eps = nn.Sequential(
            nn.Linear(in_dim + z_dim, self.hidden_dim_enc),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_enc,out_dim)
        )

        self.sigma = nn.Sequential(
            nn.Parameter(torch.rand(z_dim)),
            nn.Softplus()
        )
        ##

        # decoder p_\phi(x|z) #
        self.p_phi = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.Relu(),
            nn.Linear(self.hidden_dim, out_dim),
            nn.Sigmoid()
        )
        ##

    def forward(self, x):
        # u_s ~ q(u)
        u = self.q_u.sample()
        
        # \epsilon_s ~ q(\epsilon)
        epsilon = self.q_eps.sample()

        # z_s = h_\theta(u_s; \epsilon_s)
        mu_eps = self.mu_eps(torch.stack([x,epsilon],dim=1))
        z = mu_eps + self.sigma*u

        return z
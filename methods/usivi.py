import torch

import torch.nn.functional as F

from .hmc import hmc

def usivi(data, model, history, indices, batch_ratio, step_delta, burn_iters, samp_iters, leap_frog, mc_samples, adapt):
    logp = 0

    for _ in range(mc_samples):
        z, epsilon = model(data)

        samples, extra_outputs = hmc(data, z, epsilon, model, step_delta, burn_iters, samp_iters, adapt, leap_frog)

        history['accept_hist'][:, indices] += batch_ratio*extra_outputs['accept_hist']/(samp_iters*mc_samples)
        history['acc_rate'][indices] += batch_ratio*extra_outputs['acc_rate']/(samp_iters*mc_samples)
        
        step_delta = extra_outputs['delta']
        Tr_epsilon_t = torch.zeros((data.size(0), z.size(1)), device=model.device)

        for j in range(samp_iters):
            epsilon_j = samples[j]

            _, eps_j = model(data,epsilon_j)
            Tr_epsilon_t += eps_j / samp_iters

        # Model component
        logpxz = model.p_x_z_logdensity(data,z)
        
        # Entropy component
        # \Delta logqz = (z.detach() - Tr_epsilon_t.detach())/(F.softplus(model.sigma.detach())**2)
        Dlogqz = (z.detach() - Tr_epsilon_t.detach())/(F.softplus(model.sigma.detach())**2)

        # Average the log-joint
        logp += logpxz/mc_samples

        # How combine Dlogpxz and Dlogqz ??

    return logp.mean().item(), None
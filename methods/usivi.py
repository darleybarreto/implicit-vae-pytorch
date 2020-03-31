import torch

import torch.nn.functional as F

from .hmc import hmc

def usivi(data, model, optimizer, split, history, indices, batch_ratio, step_delta, burn_iters, samp_iters, leap_frog, mc_samples, adapt):
    logp = 0
    losses = 0

    for _ in range(mc_samples):
        z, epsilon = model(data)

        samples, extra_outputs = hmc(data, z, epsilon, model, step_delta, burn_iters, samp_iters, adapt, leap_frog)

        history['accept_hist'][:, indices] += batch_ratio*extra_outputs['accept_hist']/(samp_iters*mc_samples)
        history['acc_rate'][indices] += batch_ratio*extra_outputs['acc_rate']/(samp_iters*mc_samples)
        
        step_delta = extra_outputs['delta']
        Tr_epsilon_t = torch.zeros((data.size(0), z.size(1)), device=model.device, requires_grad=True)

        # for j in range(samp_iters):
        #     epsilon_j = samples[j]

        #     _, eps_j = model(data,epsilon_j)
        #     Tr_epsilon_t = Tr_epsilon_t + eps_j / samp_iters

        data_expd = data.unsqueeze(dim=0).repeat(samp_iters, 1, 1)
        _, eps_samples = model(data_expd,samples)
        Tr_epsilon_t = eps_samples.mean(dim=0)

        # Model component
        logpxz = model.p_x_z_logdensity(data,z).mean()
        
        # Entropy component
        logqz = torch.sum(0.5*(((z - Tr_epsilon_t)/model.sigma)**2),dim=1).mean()

        # Perform backward pass
        loss = -(logpxz - logqz)

        if 'Training' == split:
            loss.backward()

            optimizer.step()

        # Final loss
        losses += loss.item()/mc_samples

        # Average the log-joint
        logp += (logpxz.item()/mc_samples)

    return logp, losses
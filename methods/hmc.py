import torch

def hmc(data, z, current_q, model, step_delta, burn, samp_iters, adapt, leap_frog):
    current_q = current_q.detach()
    z = z.detach()

    z = z
    N, n = current_q.shape

    accept_hist = torch.zeros((burn+samp_iters, N),device=model.device) 
    logpxz_hist = torch.zeros((burn+samp_iters, N),device=model.device)
    
    #collected samples 
    samples = torch.zeros((samp_iters, N, n), device = model.device,requires_grad = True) 
    extra_outputs = {}

    if burn+samp_iters == 0:
        z = current_q
        
        extra_outputs['delta'] = step_delta
        extra_outputs['acc_rate'] = 0

        return z, samples, extra_outputs

    eta = 0.01
    opt = 0.9
    cnt = 0 

    for i in range(burn + samp_iters):
        # compute the hamiltonian
        q = current_q                                   # initial position
        p = torch.randn((N,n),device=model.device)      # sample initial momentum

        current_p = p

        # Make a half step for momentum at the beginning
        logpxz, gradz = model.q_z_e_logdensity(data,z,q)
        current_U = - logpxz # initial potential energy
        
        # step one of the leap frog integrator updating the momentum
        grad_U = - gradz
        p = p - step_delta*grad_U/2

        # Alternate full steps for position and momentum
        for j in range(leap_frog):
            # Make a full step for the position
            q = q + step_delta*p

            #Make a full step for the momentum, except at end of trajectory
            
            if j < (leap_frog-1):
                logpxz, gradz = model.q_z_e_logdensity(data,z,q) 
                proposed_U = - logpxz
                grad_U = - gradz
                p = p - step_delta*grad_U

        # Make a half step for momentum at the end.
        logpxz, gradz = model.q_z_e_logdensity(data,z,q)
        proposed_U = - logpxz
        grad_U = - gradz
        p = p - step_delta*grad_U/2
        
        #Negate momentum at end of trajectory to make the proposal symmetric
        p = -p

        # Evaluate potential and kinetic energies at start and end of trajectory
        current_K = torch.sum(current_p**2, 1)/2
        proposed_K = torch.sum(p**2, 1)/2 

        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        accept = torch.rand(N,device=model.device) < torch.exp(current_U-proposed_U+current_K-proposed_K)
        
        accept_hist[i] = accept.long()

        ind = accept==1

        if len(ind) > 0:
            current_q[ind,:] = q[ind,:]
            current_U[ind] = proposed_U[ind]

        # Adapt step size only during burn-in. After that
        # collect samples  
        if i <= burn and adapt == 1:
            step_delta = step_delta + eta*((torch.mean(accept.float()) - opt)/opt)*step_delta
        
        else:
            cnt = cnt + 1
            samples[cnt] = current_q
        
        
        logpxz_hist[i] = - current_U

    extra_outputs['logpxz_hist'] = logpxz_hist
    extra_outputs['accept_hist'] = accept_hist
    extra_outputs['delta'] = step_delta 
    extra_outputs['acc_rate'] = torch.mean(accept_hist, dim=0)

    return samples.detach(), extra_outputs
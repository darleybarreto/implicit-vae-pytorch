import torch

def hmc(current_q, log_pxz, epsilon, burn, T, adapt, L):
    N, n = current_q.shape

    accept_hist = torch.zeros(N, burn+T) 
    logpxz_hist = torch.zeros(N, burn+T)
    
    #collected samples 
    samples = torch.zeros(N, n, T) 
    extra_outputs = {}

    if burn+T == 0:
        z = current_q
        
        extra_outputs['delta'] = epsilon
        extra_outputs['accRate'] = 0

        return z, samples, extra_outputs

    eta = 0.01
    opt = 0.9
    cnt = 0 

    for i in range(burn + T):
        q = current_q
        p = torch.randn(N,n)

        current_p = p

        # Make a half step for momentum at the beginning
        logpxz, gradz = log_pxz.logdensity(q)
        current_U = - logpxz
        grad_U = - gradz
        p = p - epsilon*grad_U/2

        # Alternate full steps for position and momentum
        for j in range(L):
            # Make a full step for the position
            q = q + epsilon*p

            #Make a full step for the momentum, except at end of trajectory
            
            if j != L:
                logpxz, gradz = log_pxz.logdensity(q) 
                proposed_U = - logpxz
                grad_U = - gradz
                p = p - epsilon*grad_U

        # Make a half step for momentum at the end.
        logpxz, gradz = log_pxz.logdensity(q)
        proposed_U = - logpxz
        grad_U = - gradz
        p = p - epsilon*grad_U/2
        
        #Negate momentum at end of trajectory to make the proposal symmetric
        p = -p

        # Evaluate potential and kinetic energies at start and end of trajectory
        current_K = torch.sum(torch.pow(current_p,2), 2)/2
        proposed_K = torch.sum(torch.pow(p,2), 2)/2 

        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        accept = (torch.rand(N,1) < torch.exp(current_U-proposed_U+current_K-proposed_K))
        
        accept_hist[:, i] = accept

        ind = accept==1
        if len(ind) > 0:
            current_q[ind,:] = q[ind,:]
            current_U[ind] = proposed_U[ind]

        # Adapt step size only during burn-in. After that
        # collect samples  
        if i <= burn and adapt == 1:
            epsilon = epsilon + eta*((torch.mean(accept) - opt)/opt)*epsilon
        
        else:
            cnt = cnt + 1
            samples[:,:,cnt] = current_q
        
        
        logpxz_hist[:,i] = - current_U

    z = current_q
    extra_outputs['logpxz_hist'] = logpxz_hist 
    extra_outputs['accept_hist'] = accept_hist
    extra_outputs['delta'] = epsilon 
    extra_outputs['accRate'] = torch.mean(accept_hist, dim=1)

    return z, samples, extra_outputs
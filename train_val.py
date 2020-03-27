import contextlib

import torch

def train_val(history,loader,method,model,device,optimizer,epoch,args,train=True):
    if args.method == 'usivi':
        extra_args = {'history':history,
                    'batch_ratio':len(loader)/args.batch_size,
                    'step_delta':0.2,
                    'adapt':1,
                    'leap_frog':5,
                    'burn_iters':args.burn,
                    'samp_iters':args.sampling,
                    'mc_samples':args.mcmc_samples}
    else:
        if epoch<1900:
            J = 1
            warm_up = min([epoch/300,1])

        else:
            J = 50
            warm_up = 1

        extra_args = {'K':args.K,"J":J,'warm_up':warm_up}
        

    stochastic_bound = []
    losses = []

    if train:
        ctx_mgr = contextlib.suppress()
        model.train()
    else:
        ctx_mgr = torch.no_grad()
        model.eval()
    
    with ctx_mgr:
        for indices, data in loader:
            data = data.to(device)
            data.requires_grad = True

            optimizer.zero_grad()

            if args.method == 'usivi':
                extra_args.update({'indices': indices})

            logp, loss = method(data, model, **extra_args)

            optimizer.step()

            stochastic_bound.append(logp)

            if not loss is None:
                losses.append(loss.item())

    return sum(stochastic_bound)/len(stochastic_bound), losses
            
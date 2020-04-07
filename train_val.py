import contextlib

import torch

def train_val(history,loader,method,model,device,optimizer,scheduler,epoch,args,train=True):
    if args.method == 'usivi':
        extra_args = {'history':history,
                    'batch_ratio':len(loader)/args.batch_size,
                    'step_delta':0.5/args.dim_z,
                    'adapt':1,
                    'leap_frog':5,
                    'burn_iters':args.burn,
                    'samp_iters':args.sampling,
                    'mc_samples':args.mcmc_samples}
    else:
        if epoch<1900:
            J = 1

        else:
            J = 50

        extra_args = {'K':args.K,"J":J}

    tot_data = len(loader)
    
    # to_log = tot_data//2
    to_log = tot_data//10

    stochastic_bound = []
    losses = []

    if train:
        split = "Training"
        ctx_mgr = contextlib.suppress()
        model.train()

    else:
        split = "Validation"
        ctx_mgr = torch.no_grad()
        model.eval()
    
    print(f"===== {split} =====")

    with ctx_mgr:
        for i, (indices, data) in enumerate(loader):
            data = data.to(device)
            data.requires_grad = True

            if args.method == 'usivi':
                extra_args.update({'indices': indices})

            logp, loss = method(data, model, optimizer,scheduler, split, **extra_args)

            stochastic_bound.append(logp)
            losses.append(loss)

            if i%to_log == 0:
                
                print(f"Epoch: {epoch+1}/{args.epoches} | Iteraton Log: {i}/{tot_data} | Loss: {loss} | Log[p(x)]: {logp}")
    
    print("===================")
                
    return stochastic_bound, losses
            
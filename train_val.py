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

    tot_data = len(loader)*args.batch_size
    to_log = tot_data//60000

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

            optimizer.zero_grad()

            if args.method == 'usivi':
                extra_args.update({'indices': indices})

            logp, loss = method(data, model, optimizer, **extra_args)

            stochastic_bound.append(logp)
            losses.append(loss)

            if i%to_log == 0:
                
                print(f"Epoch: {epoch}/{args.epoches} | Iteraton Log: {i}/{to_log} | Loss: {loss} | Log[p(x)]: {logp}")
    
    print("===================")
                
    return sum(stochastic_bound)/len(stochastic_bound), losses
            
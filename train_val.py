def train_val(loader,method,model,device,optimizer,epoch,args,train=False):
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
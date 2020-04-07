import argparse
from statistics import mean

from torchvision import transforms
from torchvision import datasets as torch_datasets

import torch
import models

from methods.usivi import usivi
from methods.sivi import sivi

from train_val import train_val

from utils import Binarize, Dataset, compute_llh_vae


parser = argparse.ArgumentParser(description='PyTorch Implementation of a couple methods on Semi-Implicit Variational Inference')
parser.add_argument('-d','--dataset', type=str, default="bmnist", choices=['bmnist', 'fashionmnist'],
					help='Indicate the dataset. It can take on one of these values: [bmnist, fashionmnist]')
parser.add_argument('-n','--method', type=str, default="usivi", choices=['sivi', 'usivi'],
					help='Specify the method. It can take on one of these values: [sivi, usivi]')
parser.add_argument('-z','--z-dim', type=int, default=0,help='Number dimension of the latent space. If none passed, defaults will be used')					
parser.add_argument('-b','--burn', type=int, default=5,help='Number of burning iterations for the HMC chain')
parser.add_argument('-s','--sampling', type=int, default=5,help='Number of samples obtained in the HMC procedure for the reverse conditional')
parser.add_argument('--mcmc-samples', type=int, default=5, metavar="MS",help='Number of samples to be drawn from HMCMC')
parser.add_argument('--batch-size', type=int, default=100, metavar="BTCH",help='Minibatch size')
parser.add_argument('-e','--epoches', type=int, default=135,help='Number of epoches to run')
parser.add_argument('-k','--K', type=int, default=1,help='number of samples for importance weight sampling')
parser.add_argument('-t','--train', action='store_true', default=False,help='If it is train or test')

methods = {'usivi':usivi,'sivi':sivi}
models = {'usivi':models.UVAE, 'sivi': models.SIVAE}

datasets = {'bmnist':torch_datasets.MNIST, 'fashionmnist': torch_datasets.FashionMNIST}

if __name__ == "__main__":
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
	kwargs.update({'shuffle':True,"drop_last":True,"batch_size":args.batch_size})

	assert args.method in methods, f"Method not found, {args.method}"
	method = methods[args.method]
	
	if args.z_dim > 0:
		model = models[args.model](in_dim=784,z_dim=args.z_dim,device=device)
	else:
		model = models[args.model](in_dim=784,device=device)

	assert args.dataset in datasets, f"Model not found, {args.dataset}"
	dataset = lambda is_train: Dataset(datasets[args.dataset]('data/', train=is_train, download=True,transform=transforms.Compose([transforms.ToTensor(), Binarize()])))

	train_loader = torch.utils.data.DataLoader(dataset(True), **kwargs)
	val_loader = torch.utils.data.DataLoader(dataset(False), **kwargs)

	model = model.to(device)

	if args.method == 'usivi':
		optimizer = torch.optim.Adam([
			{'params':model.mu_eps.parameters(), 'lr':1e-3},
			{'params':model.sigma, 'lr':2*1e-4},
			{'params':model.p_phi.parameters(),'lr':1e-3},
		])
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3000, gamma=0.9)

	else:
		optimizer = torch.optim.Adam(model.parameters())
		scheduler = None
	
	history = {}
	for split in ['train', 'val']:
		history[split] = {'accept_hist' : torch.zeros(args.burn+args.mcmc_samples, len(train_loader)*args.batch_size).to(device),\
						  'acc_rate' : torch.zeros(len(train_loader)*args.batch_size).to(device)}

	llh_test = []
	pxs = []
	elbo = []
	
	for epoch in range(args.epoches):
		m_s_b_train, losses_t = train_val(history['train'],train_loader,method,model,device,optimizer,scheduler,epoch,args)
		
		print("Mean Stochastic Bound train:",mean(m_s_b_train))
		print(f"Value of mean DKL on train dataset for epoch {epoch} is {mean(losses_t)}")

		val_opt = 'mean DKL'

		if args.method == 'usivi':
			if epoch == args.epoches - 1:
				T = 10000
				S = 1000

			else:
				T = 500
				S = 100

			if epoch%30 == 0 or epoch == args.epoches - 1:
				logjoint, marginal = compute_llh_vae(T, S, model, val_loader)
				llh_test.append(marginal)

				mean_l = mean(marginal)
				val_opt = 'marginal'

			else:
				logjoint, elbo = train_val(history['val'],val_loader,method,model,device,optimizer,scheduler,epoch,args,train=False)
				mean_l = mean(elbo)

		else:
			logjoint, elbo = train_val(history['val'],val_loader,method,model,device,optimizer,scheduler,epoch,args,train=False)
			mean_l = mean(elbo)

		m_joint = mean(logjoint)

		print("Mean Stochastic Bound val:",m_joint)
		print(f"Value of {val_opt} on val dataset for epoch {epoch} is {mean_l}")
		print()

		if val_opt != 'marginal':
			elbo.append(mean_l)

		pxs.append(m_joint)

	best_elbo = min(elbo)
	idx = elbo.index(best_elbo)
	print(f"The best val Loss was {best_elbo} at epoch {idx}")

	best_pxs = max(pxs)
	idx = pxs.index(best_pxs)
	print(f"The best val log likelihood was {best_pxs} at epoch {idx}")

	if len(llh_test) > 0:
		print(10*'=')
		print(f"{5*'='} USIVI {5*'='}")

		print(f"Marginal on {len(llh_test)} epochs is", mean(llh_test))

		best_llh = max(llh_test)
		idx = llh_test.index(best_llh)

		print(f"The best marginal was {best_llh} at {idx}")
		print(10*'=')
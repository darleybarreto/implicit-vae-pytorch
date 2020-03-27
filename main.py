import argparse

from torchvision import transforms
from torchvision import datasets as torch_datasets

import torch
import models

from methods.usivi import usivi
from methods.sivi import sivi

from train_val import train_val

from utils import Binarize, Dataset, compute_llh_vae


parser = argparse.ArgumentParser(description='PyTorch Implementation of Unbiased Implicit Variational Inference (UIVI')
parser.add_argument('-m','--model', type=str, default="uvae", choices=['uvae','vae'],
					help='Specify the model. It can take on one of these values: [uvae,vae]')
parser.add_argument('-d','--dataset', type=str, default="bmnist", choices=['bmnist', 'fashionmnist'],
					help='Indicate the dataset. It can take on one of these values: [bmnist, fashionmnist]')
parser.add_argument('-n','--method', type=str, default="usivi", choices=['sivi', 'usivi'],
					help='Specify the method. It can take on one of these values: [sivi, usivi]')
parser.add_argument('-b','--burn', type=int, default=5,help='Number of burning iterations for the HMC chain')
parser.add_argument('-s','--sampling', type=int, default=5,help='Number of samples obtained in the HMC procedure for the reverse conditional')
parser.add_argument('--mcmc-samples', type=int, default=5, metavar="MS",help='Number of samples to be drawn from HMCMC')
parser.add_argument('--batch-size', type=int, default=100, metavar="BTCH",help='Minibatch size')
parser.add_argument('-e','--epoches', type=int, default=100,help='Number of epoches to run')
parser.add_argument('-k','--K', type=int, default=1,help='number of samples for importance weight sampling')
parser.add_argument('-t','--train', action='store_true', default=False,help='If it is train or test')

methods = {'usivi':usivi,'sivi':sivi}
models = {'uvae':models.UVAE}

datasets = {'bmnist':torch_datasets.MNIST, 'fashionmnist': torch_datasets.FashionMNIST}

if __name__ == "__main__":
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {}
	kwargs.update({'shuffle':True,"drop_last":True,"batch_size":args.batch_size})

	assert args.method in methods, f"Method not found, {args.method}"
	method = methods[args.method]

	assert args.model in models, f"Model not found, {args.model}"
	model = models[args.model](in_dim=784,device=device)

	assert args.dataset in datasets, f"Model not found, {args.dataset}"
	dataset = lambda is_train: Dataset(datasets[args.dataset]('data/', train=is_train, download=True,transform=transforms.Compose([transforms.ToTensor(), Binarize()])))

	train_loader = torch.utils.data.DataLoader(dataset(True), **kwargs)
	val_loader = torch.utils.data.DataLoader(dataset(False), **kwargs)

	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters())
	
	history = {}
	for split in ['train', 'val']:
		history[split] = {'accept_hist' : torch.zeros(args.burn+args.mcmc_samples, len(train_loader)*args.batch_size).to(device),\
						  'acc_rate' : torch.zeros(len(train_loader)*args.batch_size).to(device)}

	llh_test = []

	for epoch in range(args.epoches):
		m_s_b_train, losses_t = train_val(history['train'],train_loader,method,model,device,optimizer,epoch,args)
		print("Mean Stochastic Bound train:",m_s_b_train)

		m_s_b_val, losses_v = mean_stochastic_bound = train_val(history['val'],val_loader,method,model,device,optimizer,epoch,args,train=False)
		print("Mean Stochastic Bound val:",m_s_b_val)

		if len(losses_v)==0 and epoch%100 == 0:
			if epoch == args.epoches - 1:
				T = 10000
				S = 1000

			else:
				T = 500
				S = 100

			mean_px = compute_llh_vae(T, S, model, val_loader)
			llh_test.append(mean_px)

			print(f"Value of mean Log Likelihood on val dataset for epoch {epoch} is {mean_px}")

		elif len(losses_v)>0 and epoch%100 == 0:
			mean_l = sum(losses_v)/len(losses_v)
			
			print(f"Value of mean DKL on val dataset for epoch {epoch} is {mean_l}")


import argparse

import models

from methods.usivi import usivi
from methods.sivi import sivi

from train_val import train_val
from test import test

from utils.auxiliary import BatchSubset, Binarize, random_batch_split

from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch Implementation of Unbiased Implicit Variational Inference (UIVI')
parser.add_argument('-m','--model', type=str, default="vae", metavar="M", choices=['vae'],
					help='Specify the model. It can take on one of these values: [vae]')
parser.add_argument('-d','--dataset', type=str, default="bmnist", metavar="D", choices=['bmnist', 'fashionmnist'],
					help='Indicate the dataset. It can take on one of these values: [bmnist, fashionmnist]')
parser.add_argument('-n','--method', type=str, default="uivi", metavar="N", choices=['sivi', 'usivi'],
					help='Specify the method. It can take on one of these values: [sivi, usivi]')
parser.add_argument('-b','--burn', type=int, default=5, metavar="B",help='Number of burning iterations for the HMC chain')
parser.add_argument('-s','--sampling', type=int, default=5, metavar="S",help='Number of samples obtained in the HMC procedure')
parser.add_argument('--batch_size', type=int, default=100, metavar="BTCH",help='Minibatch size')
parser.add_argument('-e','--epoches', type=int, default=100, metavar="EPCH",help='Number of epoches to run')
parser.add_argument('-f','--train-frac', type=float, default=0.8, metavar="FRAC",help='Fraction of train set and val')
parser.add_argument('-t','--train', action='store_true', default=False, metavar="ISTR",help='If it is train or test')

methods = {'usivi':usivi,'sivi':sivi}
models = {'vae':models.imp_mixture.ImpMix}

datasets = {
	'bmnist': lambda is_train: datasets.MNIST('data/', train=is_train, download=True,
        						transform=transforms.Compose([transforms.ToTensor(), Binarize()])),
	'fashionmnist': lambda is_train: datasets.FashionMNIST('data/', train=is_train, download=True,
        						transform=transforms.Compose([transforms.ToTensor(), Binarize()])),
}

if __name__ == "__main__":
	args = parser.parse_args()

	assert args.n in methods, f"Method not found, {args.n}"
	method = methods[args.n]

	assert args.m in models, f"Model not found, {args.m}"
	model = models[args.m]

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

	assert args.d in datasets, f"Model not found, {args.d}"
	dataset = datasets[args.d](args.t)

	if args.t:
		batches_len = len(dataset)//args.batch_size

		train_indices, val_indices = \
			random_batch_split(dataset, args.batch_size, \
			[int(batches_len*args.f), math.ceil(batches_len*( 1 - args.f))])

		train_dataset = BatchSubset(dataset,train_indices)
		val_dataset = BatchSubset(dataset,val_indices)

		train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)

		optimizer = torch.optim.Adam(model.parameters())

		for epoch in range(args.e):
			model.train()
			train_val(train_loader,method,model,device,optimizer,epoch,args,train=True)

			with torch.no_grad():
				model.eval()
				train_val(val_loader,method,model,device,optimizer,epoch,args)

	else:
		test()
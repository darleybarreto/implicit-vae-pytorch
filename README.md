## Implicit-Vae-Pytorch

This repository has two implementations of Semi-Implicit Variational Autoencoders (not finished yet):
1. The original [Semi-Implicit Variational Inference](https://arxiv.org/pdf/1805.11183.pdf) paper and offitial [github](https://github.com/mingzhang-yin/SIVI) repo I used to reimplement.
2. [Unbiased Implicit Variational Inference](https://arxiv.org/pdf/1808.02078.pdf) and offitial [github](https://github.com/franrruiz/uiviI) repo I used to reimplement.  

### Usage
```
$ python3 main.py
  -h, --help            show this help message and exit
  -m {uvae,svae}, --model {uvae,svae}
                        Specify the model. It can take on one of these values:
                        [uvae,svae]
  -d {bmnist,fashionmnist}, --dataset {bmnist,fashionmnist}
                        Indicate the dataset. It can take on one of these
                        values: [bmnist, fashionmnist]
  -n {sivi,usivi}, --method {sivi,usivi}
                        Specify the method. It can take on one of these
                        values: [sivi, usivi]
  -b BURN, --burn BURN  Number of burning iterations for the HMC chain
  -s SAMPLING, --sampling SAMPLING
                        Number of samples obtained in the HMC procedure for
                        the reverse conditional
  --mcmc-samples MS     Number of samples to be drawn from HMCMC
  --batch-size BTCH     Minibatch size
  -e EPOCHES, --epoches EPOCHES
                        Number of epoches to run
  -k K, --K K           number of samples for importance weight sampling
  -t, --train           If it is train or test
```

### Dependencies
* numpy >= 1.17
* pytorch >= 1.4

### Results (on MNIST only)
1. Training with batch size of 135 and 2000 epochs, the lowest variational bound was 133.39 at epoch 5 and the biggest log likelihood of -76.34 at epoch 1747.  
2. None yet.  

### Problems
2. Implementation of `2.` not functional (weird gradient values).  
<div align="center">

# Implementation of Various Autoencoders

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

<div align="left">

## Autoencoder

**Learning representations by back-propagating errors**. Nature 1986.

>We describe a new learning procedure, back-propagation, for networks of neurone-like units. The procedure repeatedly adjusts the weights of the connections in the network so as to minimize a measure of the difference between the actual output vector of the net and the desired output vector. As a result of the weight adjustments, internal ‘hidden’ units which are not part of the input or output come to represent important features of the task domain, and the regularities in the task are captured by the interactions of these units. The ability to create useful new features distinguishes back-propagation from earlier, simpler methods such as the perceptron-convergence procedure1.

## Variational Autoencoder

**Auto-Encoding Variational Bayes**. ICRL 2014.

>Can we efficiently learn the parameters of directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions? We introduce an unsupervised on-line learning method that efficiently optimizes the variational lower bound on the marginal likelihood and that, under some mild conditions, even works in the intractable case. The method optimizes a probabilistic encoder (also called a recognition network) to approximate the intractable posterior distribution of the latent variables. The crucial element is a reparameterization of the variational bound with an independent noise variable, yielding a stochastic objective function which can be jointly optimized w.r.t. variational and generative parameters using standard gradient-based stochastic optimization methods. Theoretical advantages are reflected in experimental results.


## Beta Variational Autoencoder

**beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**. ICLR 2017.

>Learning an interpretable factorised representation of the independent data generative factors of the world without supervision is an important precursor for the development of artificial intelligence that is able to learn and reason in the same way that humans do. We introduce beta-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner. Our approach is a modification of the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter beta that balances latent channel capacity and independence constraints with reconstruction accuracy. We demonstrate that beta-VAE with appropriately tuned beta > 1 qualitatively outperforms VAE (beta = 1), as well as state of the art unsupervised (InfoGAN) and semi-supervised (DC-IGN) approaches to disentangled factor learning on a variety of datasets (celebA, faces and chairs). Furthermore, we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models, and show that our approach also significantly outperforms all baselines quantitatively. Unlike InfoGAN, beta-VAE is stable to train, makes few assumptions about the data and relies on tuning a single hyperparameter, which can be directly optimised through a hyper parameter search using weakly labelled data or through heuristic visual inspection for purely unsupervised data.
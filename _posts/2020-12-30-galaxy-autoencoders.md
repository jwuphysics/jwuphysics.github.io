---
title: 'What do galaxies look like? Learning from variational autoencoders'
date: 2020-12-30
permalink: /blog/2020/12/galaxy-autoencoders/
tags:
  - galaxies
  - tutorial
  - substack-blog
---

Exploring the latent space of galaxy images with autoencoders.

## Prelude

How much information do we actually use from astronomical imaging? I’ve always wondered about this as I [gaze at wide-field survey imaging data](https://www.legacysurvey.org/viewer/). After all, galaxies are complex collections of stars, gas, and dust, but for convenience, astronomers often represent them using just a few numbers that describe their colors, shapes, and sizes. Seems a bit unfair, right?

![A glimpse of NGC 3769, UGC 6576, and many other galaxies from DECaLS in the Legacy Surveys image viewer.]({{ site.baseurl }}/images/blog/decals-examples.png)

It’s for this reason that I got interested in using machine learning to utilize *all* the information that’s available in astronomical images—down to the pixel scale. (You can read about my first foray into deep learning in a [previous post](https://jwuphysics.github.io/blog/2020/05/exploring-galaxies-with-deep-learning/)). As machine learning methods continue to grow more popular in the astronomical research community, it seems more and more apparent that we should be using convolutional neural networks (CNNs) or other computer vision techniques to process galaxies’ morphological information. These models are flexible enough to make sense of diverse galaxy appearances, and connect them to the physics of galaxy evolution.

Up until now, however, I’ve mostly used CNNs to make predictions—a form of *supervised* machine learning. There are other *unsupervised* or *semi-supervised* methods, which generally try to teach a model to learn about the underlying structure of the data. For example, we might expect that a model should be able to figure out that the same galaxy can be positioned at different angles or inclinations, and that it could supply a few variables to encode the orientation of the system. Unsupervised machine learning models can figure out these kinds of patterns without explicitly being taught the position angle or inclination.

So I set out to experiment a bit with some unsupervised models (specifically *autoencoders*—more on that later). But first, I needed a good data set!

## Stumbling across a neat data set

I had the pleasure of attending the [NeurIPS 2020 conference](https://neurips.cc/) and [presenting some recent work during one of the workshops](https://ml4physicalsciences.github.io/2020/) three weeks ago. During one the talk/tutorial sessions, I found out about a neat little dataset called [Galaxy10](https://astronn.readthedocs.io/en/latest/galaxy10.html), which is a collection of 21,000 galaxy images and their morphological classifications from [GalaxyZoo](https://data.galaxyzoo.org/) (which, by the way, is a citizen science project). Galaxy10 is meant to be analogous to the well-known [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) data set frequently used in machine learning.

![]({{ site.baseurl }}/images/blog/galaxy10_example.png)

Since these images are extremely morphologically diverse, and the data set seemed to be free from corrupted images and other artifacts, I decided to train an autoencoder to reconstruct these images.

*(If you want to jump straight to the code, take a look at my [Github repository](https://github.com/jwuphysics/galaxy-autoencoders).)*

## What is an autoencoder?

In short, an autoencoder is a type of neural network that takes some input, encodes it using a small amount of information (i.e., a few *latent variables*), and then decodes it in order to recreate the original information. For example, we might take a galaxy image that has 3 channels and 69×69 pixels (for a total of 14,283 pieces of information), and then pass it through several convolution layers so that it can be represented with only 50 neurons, and then pass it through some transposed convolutions in order to recover the original 3×69×69 pixel image. It is considered an unsupervised learning model because the training input and target are the same image—no other labels are needed!

![Credit: https://www.compthree.com/blog/autoencoder/]({{ site.baseurl }}/images/blog/vae-diagram.png)

An autoencoder is essentially a compression algorithm facilitated by neural networks: the encoder and decoder are learned via data examples and gradient descent/backpropagation. Another way to think about it is a non-linear (and non-orthogonal) version of principal components analysis.

I decided to use a [variational autoencoder (VAE)](https://arxiv.org/abs/1312.6114), which probabilistically reconstructs the inputs by *sampling* from the latent space. That is, the model learns a set of latent variable *means* and *variances*, from which it can construct a multivariate normal distribution. It will then select points near the encoded latent variable and compare decoded outputs to the original inputs; this ensures that the latent space is smoothly varying. There is a lot more to variational inference than what I’ve written here, so please feel free to check out other resources on the matter (e.g., [this blog post](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)). While training VAEs, we often find that the latent space contains interpretable information about the structure of the inputs.

## Exploring VAE loss functions

I experimented with a few different terms in the VAE loss function: *evidence lower bound* (ELBO), [*maximum mean discrepancy*](https://proceedings.neurips.cc/paper/2006/file/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Paper.pdf)[ (MMD)](https://proceedings.neurips.cc/paper/2006/file/e9fb2eda3d9c55a0d89c98d6c54b5b3e-Paper.pdf), [perceptual loss](https://cs.stanford.edu/people/jcjohns/eccv16/), and class-aware cross-entropy loss.

All loss functions should penalize the pixel reconstruction error, so I imposed a mean squared error (MSE) penalty for differences between the input and output pixel values. In essence, the reconstruction error makes sure that every pixel in the output image actually looks like the corresponding one in the input image. ELBO comprises the MSE and a Kullback–Leibler (KL) divergence term. The KL divergence measures the dissimilarity between the multivariate normal distribution (a simplifying assumption that we made as part of our VAE) and the true underlying probability distribution of a latent variable given some input.

However, optimizing a VAE using ELBO tends to blur out information that isn’t well-represented in the input examples; this is a common problem with vanilla VAE models. See [this explanation](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/), which does a great job of explaining why we need additional loss terms to rectify this problem. MMD is one such term that penalizes the level of dissimilarity between the *moments* of the learned and underlying distributions. Other loss terms like perceptual loss or class-aware loss may help enforce certain inductive biases that make reconstructions more expressive or informative. However, based on my limited experience, the MMD term used in information-maximizing VAEs ([InfoVAEs](https://arxiv.org/abs/1706.02262)) is sufficient to get excellent results.

![Original input images in the Galaxy10 (SDSS) data set
]({{ site.baseurl }}/images/blog/galaxy10-original.png)

![VAE reconstructions optimized using the evidence lower bound (ELBO)
]({{ site.baseurl }}/images/blog/galaxy10-elbo-reconstruction.png)

![VAE reconstructions optimized using maximum mean discrepancy (MMD)
]({{ site.baseurl }}/images/blog/galaxy10-mmd-reconstruction.png)

In this comparison, I show six input images (top), reconstructions from a VAE trained with ELBO for 50 epochs (middle), and reconstructions from a VAE trained with MMD for 50 epochs (bottom). We can see that the MMD term enables reconstructions with more detailed features, such as spiral arms, non-axisymmetric light distributions, and multiple sources. In other words, the MMD-VAE is able to encode galaxy images with higher fidelity than the ELBO-VAE.

Of course, the level of detail can still be improved. We have only used a small subsample of galaxies from the Sloan Digital Sky Survey, and of course we might get nicer results if we used a survey telescope with higher-resolution imaging. Another way to produce a higher level of reconstruction detail is to use [*generative adversarial networks*](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)[ (GANs)](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) or [*flow-based generative models*](https://openai.com/blog/glow/). But let’s save those for another post.

## Using the VAE decoder as a generator

Once we have trained a VAE, we can now interpolate across examples in the latent space and reconstruct outputs from these vectors. What do they mean?

Making neural networks hallucinate is always profitable. (See, e.g., awesome blog post by [Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) and sweet demos from [OpenAI](https://openai.com/blog/), and [distill.pub](https://distill.pub/).) So I decided to take a journey through the imaging latent space, decode the outputs, and animate them in GIF format:

![Decoded samples in the latent space with a ELBO-VAE
]({{ site.baseurl }}/images/blog/elbo-vae-latent-space.gif)

![Decoded samples from the latent space with a MMD-VAE
]({{ site.baseurl }}/images/blog/mmd-vae-latent-space.gif)

Pretty neat huh? It’s quite interesting to see how ELBO encourages the model to learn broad-brush features, which are mostly those that can be easily represented by classical morphological parameters such as concentration, ellipticity, etc. The MMD variant learns way more complicated features!

We can also just directly decode each vector as well in order to see the image feature that correlates with each latent variable. Here I’m stepping through the latent space of the ELBO-VAE. (Note that *N*σ refers to the number of standard deviations from the mean in each dimension of the latent space).

![]({{ site.baseurl }}/images/blog/elbo-vae-latent-variables.png)

Imagine all this with incredibly high-resolution imaging! Now I really can’t wait for the [*Nancy Grace Roman Space Telescope*](https://www.stsci.edu/roman) to launch.

---

All of the code to make the figures in this notebook can be found in my [Github repository](https://github.com/jwuphysics/galaxy-autoencoders). Questions? Violent objections? Send them my way on [Twitter](https://twitter.com/jwuphysics) or [email](mailto:jowu@stsci.edu). Or subscribe to see my next post!

---

This post was migrated from Substack to my blog on 2025-04-23
{: .notice}

---
title: "Stabilizing Deep Neural Networks"
permalink: /posts/2026/01/stabilizing-deep-neural-networks/
date: 2026-01-24
tags:
  - machine-learning
  - tutorial
---

Deep neural networks can be thought of as compositions of many simple transformations, each represented by a layer with trainable parameters. When the number of layers is large, the effect of multiplying many random matrices becomes exponentially unstable — they can grow or shrink exponentially. This is the primary reason that naive initialization leads to exploding or vanishing signals for both forward (activations) and backward (gradients). Nonetheless, stability is possible when each layer is close to the identity operation. With the right scaling of weights at initialization, a deep network acts like a time-discretized flow, and the total transformation resembles a matrix exponential of small perturbations. 

Earlier this month I gave a [talk](https://science.nasa.gov/astrophysics/programs/cosmic-origins/community/ai-ml-stig-lecture-series-12-jan-2026/) and tutorial on Inductive Biases to the [NASA AI/ML Science & Technology Interest Group](https://science.nasa.gov/astrophysics/programs/cosmic-origins/community/artificial-intelligence-machine-learning-science-technology-interest-group-ai-ml-stig/). Some audience members asked questions and pursued follow-up discussion about initialization, residual layers, and connections to differential equations. This post attempts to summarize the most important points and connect the dots. 

*Thanks to Gemini 3 for help with copyediting review and blog post formatting.*

## Products of random matrices and instability

Let's begin with a purely linear \\( L \\)-layer network

$$ x_L = W_{L-1} W_{L-2}\cdots W_1\, x_0, $$

where the \\( W_\ell \\) are random matrices.

Classical results in random matrix theory show that (under reasonable assumptions) the norm of this product grows or decays exponentially with the number of factors. More precisely, if the \\( \{W_\ell\} \\) are i.i.d. with suitable integrability and irreducibility conditions, then the limit

$$ \lambda_1 = \lim_{L\to\infty} \frac{1}{L} \log \left\| W_{L-1}\cdots W_1 \right\| $$

is typically nonzero. The number \\( \lambda_1 \\) is the top Lyapunov exponent. If \\( \lambda_1>0 \\), forward signals explode; if \\( \lambda_1<0 \\), they vanish. Backpropagated gradients obey the same kind of product dynamics (with transposes), so the same instability can jeopardize training as well.

This instability is not a quirk of linear networks alone. Nonlinear activations appear between matrices, but at initialization most common activations behave approximately linearly around zero (see below). Thus, without special care, both activations and gradients are driven toward regimes where numerical stability is quickly lost as depth grows.

## Avoiding instability by staying close to the identity

The way around exponential blow-up or decay is by ensuring that each layer is close to the identity transformation. To see how this works, let's express a single layer via:

$$ W_\ell = I + \varepsilon A_\ell, $$

where \\( \varepsilon>0  \\) is small and \\( A_\ell  \\) is a random matrix with mean near zero and bounded moments.

Consider the product

$$ M_L = \prod_{\ell=1}^{L} \left( I + \varepsilon A_\ell \right), $$

where the product is ordered so that \\( \ell=1 \\) acts first on the input. We can look at two regimes in more detail to gain intuition:

1. If we choose \\( \varepsilon = L^{-1} \\) and we let \\( L\to\infty \\), then—even with noncommuting operators, we can use the [Trotter product formula](https://en.wikipedia.org/wiki/Lie_product_formula) to find

$$ M_L \to \exp\!\left( \frac{1}{L}\sum_{\ell=1}^{L} A_\ell \right) \quad \text{as } L\to\infty, $$

for random \\( A_\ell \\). The average of the \\( A_\ell \\) should be finite, and thus the limit is a well-defined matrix exponential. Intuitively, we expect that many small, nearly commuting perturbations behave as a smooth exponential flow.

2. If the perturbations are larger, say \\( \varepsilon = L^{-1/2} \\), then the product converges in distribution to the [stochastic or time-ordered exponential of a matrix-valued Brownian motion](https://lpetrov.cc/rmt25/rmt25-notes/rmt2025-l10.pdf). This is a random element of the [general linear group](https://en.wikipedia.org/wiki/General_linear_group), but still avoids the exponential blow-up that we'd worry about from unscaled random products.

Either way, we see that the result is a controlled product because each factor is a small deviation from \\( I \\).

## Initialization near the identity

If you've ever taken the [Fastai Practical Deep Learning courses](https://course.fast.ai/Lessons/lesson17.html), then you'll know that good initializations in deep learning are crticial for stabilizing the forward and backward signals. We accomplish is possible by intializing each layer to be a small perturbation of the identity. The usual rule of thumb is to choose the entries of \\( W_\ell \\) to be independent, mean-zero, and with variance matching the inverse of each layer's input dimension (i.e., fan-in dimension).

We can investigate the pre-activation of neuron \\( i \\) in layer \\( \ell \\),

$$ z_i^{(\ell)} = \sum_{j=1}^{n} W_{ij}^{(\ell)}\, a_j^{(\ell-1)}, $$

where \\( a^{(\ell-1)} \\) are activations from the previous layer and \\( n \\) is the fan-in. If \\( a_j^{(\ell-1)} \\) are centered with variance near unity, and the weights have variance \\( \sigma^2 \\), then,

$$ \mathrm{Var}\!\left(z_i^{(\ell)}\right) = n\,\sigma^2\,\mathrm{Var}\!\left(a^{(\ell-1)}\right) \approx n\sigma^2. $$

To keep the scale from changing across layers at initialization, we can set \\( n\sigma^2 \approx 1 \\) for linear or tanh-like activations near zero, so \\( \sigma^2 \approx 1/n \\). For ReLU, which zeros out about half of its inputs and rescales the variance, we use \\( \sigma^2 \approx 2/n \\). These choices keep the variance of pre-activations and activations roughly constant with depth.

There is a geometric way to interpret this. When \\( n\sigma^2 \\) is of order one, the operator norm of \\( W_\ell \\) is typically of order one as well, so \\( W_\ell \\) does not significantly expand or contract the input space at initialization. In high dimensions, with mean-zero, light-tailed weights, this makes \\( W_\ell \\) act like the identity plus a small, approximately Gaussian perturbation.

## Backward stability and gradients

Backpropagation propagates gradients through transposed weights. If the forward pass is stable at initialization, then the variance of gradients with respect to activations is also preserved layer to layer, provided the same scaling is used. So if \\( \delta^{(\ell)} \\) denotes the gradient signal at layer \\( \ell \\) and \\( \phi \\) represents the element-wise activation function (such that activations \\( a^{(\ell)} = \phi (z^{\ell)}) \\)), then

$$ \delta^{(\ell)} = (W_\ell)^\top \left(\phi'(z^{(\ell)}) \odot \delta^{(\ell+1)}\right), $$

and, under independence and small-perturbation assumptions, the variance of \\( \delta^{(\ell)} \\) matches that of \\( \delta^{(\ell+1)} \\) when \\( \sigma^2 \\) is chosen as above and \\( \phi' \\) has stable variance near initialization. The gradient with respect to a weight element is a product \\( x\,\delta \\), so its typical scale is controlled once the forward and backward variances are controlled. Thus the same near-identity reasoning stabilizes both directions of signal flow.

## Why does this still work with nonlinear activations?

Common activation functions behave approximately linearly around the origin. At initialization, pre-activations are centered and have controlled variance, so the network operates near this linear regime. As a result, the variance-propagation calculations, which are exact for linear activations, remain accurate approximations. For ReLU, we can account for the gating effect by adjusting the weight variance by a factor of two. For smooth activations like tanh or GELU, we can similarly compute the derivative near zero to set the appropriate scaling. In each case, the upshot is that each layer maps inputs to outputs without gross expansion or contraction, and therefore remains in the near-identity regime.

## Why do resnets work so well?

The incredible success of [residual neural networks](https://arxiv.org/abs/1512.03385) (resnets) shows that our arguments about staying close to the identity works in practice. A residual block updates via

$$ x_{\ell+1} = x_\ell + f_\ell(x_\ell), $$

which is just an identity operation plus a small learned perturbation (in a discrete form). As depth grows, the network approximates a continuous-time flow described by an ordinary differential equation. That is, the sequence of layers behaves like a discretization of a continuous-time system,

$$ \frac{dx(t)}{dt} = f(t, x(t)). $$

To understand how small perturbations to the input evolve through such a system, we can linearize each update. Over a short step of size \\( dt \\), a perturbation is transformed by a matrix of the form

$$ I + J(t)\,dt, \qquad J(t) = \frac{\partial f(t, x(t))}{\partial x}, $$

i.e. a near-identity matrix. The full effect of many such steps is therefore a product

$$ \left(I + J(t_L)\,dt\right)\cdots \left(I + J(t_1)\,dt\right). $$

As the step size goes to zero, this product converges to the time-ordered (\\( \mathcal{T} \\)) exponential

$$ \mathcal{T}\exp\!\left( \int_0^1 J(t)\,dt \right). $$
The overall transformation is then a time-ordered exponential of the accumulated Jacobians, which mirrors the earlier product-of-near-identity-matrices viewpoint.

## Summary

This expository post tries to justify the practical recipes for initialization and explains why they are effective in keeping very deep neural networks numerically stable. Proper initialization is vital for avoiding the exponential growth or decay associated with products of random matrices. By choosing weights so that each layer is close to the identity, we ensure that the network behaves like a controlled exponential of small perturbations instead of an unstable sequence of far-from-identity operations. Variance-preserving initialization aligns forward activations and backward gradients to have stable scale across layers, while common nonlinearities can be incorporated by modest adjustments to the variance. Residual architectures explicitly encode this principle into the design, and thereby connect deep networks to the theory of continuous flows. 

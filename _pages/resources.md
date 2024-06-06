---
permalink: /resources/
title: "Resources"
excerpt: "Resources"
author_profile: true
redirect_from: 
  - /resources.html
---

## Tutorials and lectures in astronomical ML

### 2022 Astro Hack Week
I presented a [two-part course](https://github.com/AstroHackWeek/AstroHackWeek2022/tree/main/day2_ml_tutorial) on astronomical machine learning during the 2022 Astro Hack Week. There are two Jupyter notebooks with examples and practice problems shown here. The [first notebook](https://colab.research.google.com/github/jwuphysics/AstroHackWeek2022/blob/main/day2_ml_tutorial/01-intro-machine-learning.ipynb) provides an introduction to machine learning using *tabular data*. The [second notebook](https://colab.research.google.com/github/jwuphysics/AstroHackWeek2022/blob/main/day2_ml_tutorial/02-deep-learning.ipynb) presents convolutional neural networks applied to *astronomical image cutouts*. 

### 2023 KITP Program
I helped coordinate a [KITP program on Data-Driven Astronomy (*galevo23*)](https://datadrivengalaxyevolution.github.io/), which featured some very nice tutorials and talks. We covered topics like simulation-based inference, GNNs, symbolic regression, probabilistic U-nets, and much more. All machine learning tutorials can be accessed on [Github](https://github.com/DataDrivenGalaxyEvolution/galevo23-tutorials/).

### 2023 LSSTC Data Science Fellowship Program
I was a guest lecturer for the 19th Session of the [LSST-DA Data Science Fellowship Program](https://lsstdiscoveryalliance.org/programs/data-science-fellowship/). My [first lecture](https://github.com/LSSTC-DSFP/Session-19/blob/main/day3/ConvolutionalNeuralNetworks.ipynb) focused on convolutional neural networks. My [second lecture](https://github.com/LSSTC-DSFP/Session-19/blob/main/day4/GraphNeuralNetworks.ipynb) introduced graph neural networks and their applications to galaxies, dark matter halos, and large scale structure in cosmological simulations.

## Hybrid CNNs with deconvolution layers
In order to predict galaxy spectra from images, I created a CNN with hybrid normalization layers. In the NeurIPS workshop paper, we found that a combination of deconvolution layers and batch normalization can greatly improve results for CNNs trained on astronomical images. Pytorch code for this hybrid CNN can be found on [my Github page](https://github.com/jwuphysics/predicting-spectra-from-images).

## Blog
I sporadically post in my [research blog](https://jwuphysics.substack.com/), which showcases some machine learning applications in astronomy. 
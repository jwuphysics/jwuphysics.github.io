---
permalink: /resources/
title: "Resources"
excerpt: "Resources: code and stuff"
author_profile: true
redirect_from: 
  - /research.html
---

## KITP Program
I recently helped coordinate a [KITP program on Data-Driven Astronomy (*galevo23*)](https://datadrivengalaxyevolution.github.io/), which featured some very nice tutorials and talks. We covered topics like simulation-based inference, GNNs, symbolic regression, probabilistic U-nets, and much more. All machine learning tutorials can be accessed on [Github](https://github.com/DataDrivenGalaxyEvolution/galevo23-tutorials/).

## A mini-course on astronomical ML
As part of the 2022 Astro Hack Week, I presented a [two-part course](https://github.com/AstroHackWeek/AstroHackWeek2022/tree/main/day2_ml_tutorial) on astronomical machine learning. Two Jupyter notebooks with examples and practice problems are presented. The first notebook provides an [introduction to machine learning using *tabular data*](https://colab.research.google.com/github/jwuphysics/AstroHackWeek2022/blob/main/day2_ml_tutorial/01-intro-machine-learning.ipynb). The second notebook presents [convolutional neural networks applied to *astronomical image cutouts*](https://colab.research.google.com/github/jwuphysics/AstroHackWeek2022/blob/main/day2_ml_tutorial/02-deep-learning.ipynb). The first notebook should take two hours to complete, and the second notebook should take about four hours to complete.

## Hybrid CNNs with deconvolution layers
In order to predict galaxy spectra from images, I created a CNN with hybrid normalization layers. In the NeurIPS workshop paper, we found that a combination of deconvolution layers and batch normalization can greatly improve results for CNNs trained on astronomical images. Pytorch code for this hybrid CNN can be found on [my Github page](https://github.com/jwuphysics/predicting-spectra-from-images).

## Blog
I used to maintain a [research blog](https://jwuphysics.substack.com/), which presented some machine learning applications in astronomy (mostly using convolutional neural networks). 
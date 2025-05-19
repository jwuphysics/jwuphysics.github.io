---
title: 'Foundation Models in Astronomy'
date: 2025-05-16
permalink: /blog/2025/05/foundation-models-in-astronomy/
tags:
  - astronomy
  - computer-vision
  - foundation-models
  - llms
---

Here's a casual introduction to foundation models and how they might impact astronomy research in the coming years. I'm writing this on the train back from New York to Baltimore, having just wrapped up the [Foundation Models in Astronomy](https://events.simonsfoundation.org/event/0aff2690-f1cb-485f-833a-429b6c7eb7ef/summary?tm=8eYg1qvbYoaoB-i3qiSGVDkdnLEYU8RX4tCGRKsTY_w) workshop at the Simons Foundation. My co-organizers and I are planning to write up a nice, more comprehensive blog post based on our workshop discussions; in the meantime, you'll just have to settle for this.

## Foundation models are here to stay

[Foundation models](https://crfm.stanford.edu/report.html) are the base pre-trained neural networks for large language models (LLMs) like ChatGPT or Claude, vision models like DALLE, and even automated speech recognition (ASR) models like the ones that automatically caption your Youtube videos. 

These models can learn representations of data that can distinguish between examples in the training dataset. However, they're not really "trained" in the usual fashion; instead, foundation models undergo *self-supervised* learning by optimizing a contrastive or generative objective. You shouldn't be surprised to learn that Lilian Weng has incredibly comprehensive blog posts on [self-supervised learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/) and specifically [contrastive learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/).

Foundation models allow a model to learn *how* your data can be represented or generated. By minimizing a contrastive loss, you task your model to create similar representations for the same example "viewed" differently (or transformed differently under a data augmentation procedure), and different representations for different data examples. If instead, you minimize a generative loss, then you task your model with figuring out whatever representations are useful for generating another patch of the image or the next word in a text corpus. I'd wager that contrastive losses lead to stronger discriminatory power, and that generative losses lead to better generative power, but don't actually have any data to support this intuition. Oh well. 

The real power of foundation models is that (1) they can map your data into semantically meaningful embedding representations and (2) help catalyze specific downstream tasks. 

### (1) The power of embedding representations

Why should you care about latent representations of your data (i.e. your embedding space)? By converting data into embedding vectors, you can use that embedding space to perform comparisons. Concretely, if your embedding space captures the semantic meanings of your dataset, then you'll be able to measure the semantic similarity of two objects (e.g. by using a cosine similarity or some other distance measure). You can even learn a joint representation across multiple "modalities" such as text and audio.

For example, we used text embeddings to compare astronomer queries against arXiv paper abstracts when we sought to [*evaluate LLMs for astronomy research*](https://arxiv.org/abs/2405.20389). By mapping both the user query and the paper abstracts into this embedding space, and storing the latter into a vector database, we could retrieve (hopefully) relevant papers on the basis of the user query. Over the course of the JHU CLSP [2024 JSALT workshop](https://www.clsp.jhu.edu/workshops/2024-jelinek-summer-workshop-on-speech-and-language-technology/), we dramatically improved the semantic similarity search pipeline and retrieval engine, which was published alongside many other cool results in the [Pathfinder paper](https://arxiv.org/abs/2408.01556) by [Kartheik Iyer](https://kartheikiyer.github.io/). Charlie O'Neill and Christine Ye were also able to extract, disentangle, and interpret semantic concepts in the astronomy and ML literature by training sparse autoencoders over these [paper embeddings](https://arxiv.org/abs/2408.00657)!

### (2) All-purpose base models for any task

Building up this semantically rich representation of your dataset also provides an excellent starting point for any other machine learning task. We can view this pre-trained foundation model as a *base* for some later *downstream* task. For example, if a foundation model has seen all kinds of real-world images, and learned to produce self-consistent representations of the semantic content within those images, then it should be able to classify bird species or segment cars and pedestrians and roads in a much more data-efficient way.

## Foundation models in astronomy

Foundation models are also becoming common across astronomy! In the past few years, we've seen foundation models trained on galaxy image cutouts (e.g., by [Hayat et al. 2020](https://arxiv.org/abs/2012.13083), [Stein et al. 2021](https://arxiv.org/abs/2110.00023), and [Smith et al. 2024](https://arxiv.org/abs/2405.14930)), stellar spectra ([Koblischke & Bovy 2024](https://arxiv.org/abs/2411.04750)), and even multiple modalities like images and spectra ([Parker & Lanusse et al. 2024](https://arxiv.org/abs/2310.03024)) or photometric and spectroscopic time series ([Zhang et al. 2024](https://arxiv.org/abs/2408.16829)). And there are many more coming soon!

A critical question remains: Are people actually using foundation models to make new discoveries? In general, the answer is no. Most citations are simply from other papers that are also releasing their own ML models. A notable exception is from Galaxy Zoo,[^1] whose Zoobot model by [Walmsley et al. 2021](https://arxiv.org/abs/2102.08414) has amassed ~200 citations leading to actual science! It remains to be seen whether current and next-generation foundation models will deliver real scientific value.

As I mentioned at the top, the workshop organizers will be writing up another blog post focusing on our discussions and how we might guide our community of astronomical ML practitioners. Stay on the lookout for that!

**Edit (2025-05-19)**: I'm including a list of foundation models in astronomy that I currently know about. There are arguably more, e.g. autoencoder variants such as [spender](https://arxiv.org/abs/2211.07890), but I'm trying to focus on large-scale foundation models that will (hopefully) be able generalize well to many tasks. Feel free to reach out if you're think I've made an egregious omission.

| Foundation Model                          | Domain                                                                 | Method                                             |
|-------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------|
| AstroCLIP ([Parker et al. 2023](https://arxiv.org/abs/2310.03024); [Github](https://github.com/PolymathicAI/AstroCLIP)) | DESI images and spectra                                | Self-supervised (contrastive)                   |
| Maven ([Zhang et al. 2024](https://arxiv.org/abs/2408.16829)) | Multi-modal time series (photometry and spectra)                | Self-supervised (time-series)                 |
| AstroM$^3$ ([Rizhko & Bloom 2024](https://arxiv.org/abs/2411.08842))   | Multi-modal time series (photometry, spectra, and metadata)       | Self-supervised (contrastive)              |
| FALCO ([Zuo et al. 2025](https://arxiv.org/abs/2504.20290))        | Kepler time-series                                                   | Self-supervised (generative)                         |
| SpectraFM ([Koblischke & Bovy 2024](https://arxiv.org/abs/2411.04750)) | Stellar spectra (synthetic & real)                                 | Self-supervised (generative)        |
| SSL for DESI Legacy Survey ([Stein et al. 2021](https://arxiv.org/abs/2110.00023); [Github](https://github.com/georgestein/ssl-legacysurvey)) | DESI Legacy Survey galaxy images                                      | Self-supervised (contrastive)                     |
| GZ-Evo ([Walmsley et al. 2022](https://arxiv.org/abs/2206.11927); [Github](https://github.com/mwalmsley/galaxy-datasets)) | Galaxy images (multiple observatories)                                           | Self-supervised (contrastive)                        |
| AstroPT ([Smith et al. 2024](https://arxiv.org/abs/2405.14930); [Github](https://github.com/Smith42/astroPT))     | DESI Legacy Survey Galaxy images                                      | Self-supervised (generative)                          |
| Radio Galaxy Zoo ([Slijepcevic et al. 2022](https://arxiv.org/abs/2204.08816)) | Radio-wavelength galaxy images                           | Self-supervised (contrastive)                    |
| SSL for Radio Interferometric Images ([Cecconello et al. 2024](https://arxiv.org/abs/2411.14078); [Github](https://github.com/dr4thmos/solo-learn-radio)) | Radio interferometric images                                        | Self-supervised (contrastive)                   |
| SSL for LOFAR ([Baron Perez et al. 2025](https://arxiv.org/abs/2503.19111)) | Radio galaxy images (LoTSS-DR2)                                  | Self-supervised (contrastive)         |

---

[^1]: Arguably, this is expanding the definition of a foundation model because it is being trained via *supervised* learning. Zoobot learns to predict vote fractions of citizen scientists' morphological classifications.
---
title: 'Training a deep CNN to learn about galaxies in 15 minutes'
date: 2020-05-26
permalink: /blog/2020/05/galaxy-cnn/
teaser: /images/blog/WB19_fig1.jpg
tags:
  - galaxies
  - vision
  - tutorial
  - original-blog
---

Let's train a deep neural network from scratch! In this post, I provide a demonstration of how to optimize a model in order to predict galaxy metallicities using images, and I discuss some tricks for speeding up training and obtaining better results. 

**Note:** This post was migrated from my old blog. 
{: .notice}


## Predicting metallicities from pictures: obtaining the data

In my [previous post](https://github.com/jwuphysics/blog/blob/master/_notebooks/2020-05-21-exploring-galaxies-with-deep-learning.ipynb), I described the problem that we now want to solve. To summarize, we want to train a convolutional neural network (CNN) to perform regression. The inputs are images of individual galaxies (although sometimes we're photobombed by other galaxies). The outputs are metallicities, \\(Z\\), which usually take on a value between 7.8 and 9.4.

The first step, of course, is to actually get the data. Galaxy images can be fetched using calls to the Sloan Digital Sky Survey (SDSS) SkyServer `getJpeg` cutout service via their [RESTful API](http://skyserver.sdss.org/dr16/en/help/docs/api.aspx#imgcutout). For instance, [this URL](http://skyserver.sdss.org/dr16/SkyserverWS/ImgCutout/getjpeg?ra=39.8486&dec=1.094&scale=1&width=224&height=224) grabs a three-channel, \\(224 \times 224\\)-pixel JPG image:
![](images/blog/sdss_example.jpg "An example galaxy at the coordinates RA = 39.8486 and Dec = 1.094")


Galaxy metallicities can be obtained from the SDSS SkyServer using a [SQL query](http://skyserver.sdss.org/dr16/en/help/docs/sql_help.aspx) and a bit of `JOIN` magic. All in all, we use 130,000 galaxies with metallicity measurements as our training + validation data set.

The code for the original published work ([Wu & Boada 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4683W/abstract)) can be found in my [Github repo](https://github.com/jwuphysics/galaxy-cnns). However, this code (from 2018) used `fastai` *version 0.7*, and I want to show an updated version using the new and improved `fastai` *version 2* codebase. Also, some of the "best practices" for deep learning and computer vision have evolved since then, so I'd like to highlight those updates as well!

## Organizing the data using the `fastai` DataBlock API

Suppose that we now have a directory full of galaxy images, and a `csv` file with the object identifier, coordinates, and metallcity for each galaxy. The `csv` table can be [read using Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html), so let's store that in a DataFrame `df`. We can take a look at five random rows of the table by calling `df.sample(5)`:


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objID</th>
      <th>ra</th>
      <th>dec</th>
      <th>metallicity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15932</th>
      <td>1237654601557999788</td>
      <td>137.603036</td>
      <td>3.508882</td>
      <td>8.819281</td>
    </tr>
    <tr>
      <th>11276</th>
      <td>1237651067353956485</td>
      <td>189.238856</td>
      <td>65.712160</td>
      <td>8.932211</td>
    </tr>
    <tr>
      <th>44391</th>
      <td>1237661070317322521</td>
      <td>139.110354</td>
      <td>10.996989</td>
      <td>8.410136</td>
    </tr>
    <tr>
      <th>121202</th>
      <td>1237671990262825066</td>
      <td>199.912000</td>
      <td>11.290724</td>
      <td>8.980943</td>
    </tr>
    <tr>
      <th>47280</th>
      <td>1237661351633486003</td>
      <td>184.796897</td>
      <td>56.327473</td>
      <td>8.891125</td>
    </tr>
  </tbody>
</table>
</div>



This means that a galaxy with `objID` 1237654601557999788 is located at RA = 137.603036 deg, Dec = 3.508882 deg, and has a metallicity of \\(Z\\) = 8.819281. Our directory structure is such that the corresponding image is stored in `{ROOT}/images/1237660634922090677.jpg`, where `ROOT` is the path to project repository. 

A tree-view from our `{ROOT}` directory might look like this:
```
.
├── data
│   └── master.csv
├── images
│   ├── 1237654601557999788.jpg
│   ├── 1237651067353956485.jpg
│   └── [...]
└── notebooks
    └── training-a-cnn.ipynb
```

We are ready to set up our `DataBlock`, which is a core fastai construct for handling data. The process is both straightforward and extremely powerful, and comprises a few steps:
- Define the inputs and outputs in the `blocks` argument
- Specify how to get your inputs (`get_x`) and outputs (`get_y`)
- Decide how to split the data into training and validation sets (`splitter`)
- Define any CPU-level transformations (`item_tfms`) and GPU-level transformations (`batch_tfms`) used for preprocessing or augmenting your data.


Before going into the details for each component, here is the code in action:


```python
dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=ColReader(['objID'], pref=f'{ROOT}/images/', suff='.jpg'),
    get_y=ColReader(['metallicity']),
    splitter=RandomSplitter(0.2),
    item_tfms=[CropPad(144), RandomCrop(112)],
    batch_tfms=aug_transforms(max_zoom=1., flip_vert=True, max_lighting=0., max_warp=0.) + [Normalize],
)
```

Okay, now let's take a look at each part.

### Input and output blocks

First, we want to make use of the handy `ImageBlock` class for handling our input images. Since we're using galaxy images in the JPG format, we can rely on the `PIL` backend of `ImageBlock` to open the images efficiently. If, for example, we instead wanted to use images in the astronomical `FITS` format, we could extend the `TensorImage` class and define the following bit of code:


```python
class FITSImage(TensorImage):
    @classmethod
    def create(cls, filename, chans=None, **kwargs) -> None:
        """Create FITS format image by using Astropy to open the file, and then 
        applying appropriate byte swaps and flips to get a Pytorch Tensor.
        """
        return cls(
            torch.from_numpy(
                astropy.io.fits.getdata(fn).byteswap().newbyteorder()
            )
            .flip(0)
            .float()
        )
    
    def show(self, ctx=None, ax=None, vmin=None, vmax=None, scale=True, title=None):
        """Plot using matplotlib or your favorite program here!"""
        pass
        
FITSImage.create = Transform(FITSImage.create) 

def FITSImageBlock(): 
    """A FITSImageBlock that can be used in the fastai DataBlock API.
    """
    return TransformBlock(partial(FITSImage.create))
```

For our task, the vanilla `ImageBlock` will suffice. 

We also want to define an output block, which will be a `RegressionBlock` for our task (note that it handles both single- and multi-variable regression). If, for another problem, we wanted to do a categorization problem, then we'd intuitively use the `CategoryBlock`. Some other examples of the DataBlock API can be found in the [documentation](http://dev.fast.ai/data.block).

We can pass in these arguments in the form of a tuple: `blocks=(ImageBlock, RegressionBlock)`.

### Input and output object getters

Next, we want to be able to access the table, `df`, which contain the columns `objID` and `metallicity`. As we've discussed above, each galaxy's `objID` can be used to access the JPG image on disk, which is stored at `{ROOT}/images/{objID}.jpg`. Fortunately, this is easy to do with the fastai `ColumnReader` method! We just have to supply it with the column name (`objID`), a prefix (`{ROOT}/images/`), and a suffix (`.jpg`); since the prefix/suffix is only used for file paths, the function knows that the file needs to be opened (rather than interpreting it as a string). So far we have:
```python
get_x=ColReader(['objID'], pref=f'{ROOT}/images/', suff='.jpg')
```

The targets are stored in `metallicity`, so we can simply fill in the `get_y` argument:
```python
get_y=ColReader(['metallicity'])
```

(At this point, we haven't yet specified that `df` is the DataFrame we're working with. The `DataBlock` object knows how to handle the input/output information, but isn't able to load it until we provide it with `df` -- that will come later!)

### Splitting the data set 

For the sake of simplicity, we'll just randomly split our data set using the aptly named `RandomSplitter` function. We can provide it with a number between 0 and 1 (corresponding to the fraction of data that will become the validation set), and also a random seed if we wish. If we want to set aside 20% of the data for validation, we can use this:

```python
splitter=RandomSplitter(0.2, seed=56)
```

### Transformations and data augmentation

Next, I'll want to determine some data augmentation transformations. These are handy for varying our image data: crops, flips, and rotations can be applied at random using fastai's `aug_transforms()` in order to dramatically expand our data set. Even though we have >100,000 unique galaxy images, our CNN model will contain millions of trainable parameters. Augmenting the data set will be especially valuable for mitigating overfitting.

Translations, rotations, and reflections to our images should not change the properties of our galaxies. However, we won't want to zoom in and out of the images, since that might impact CNN's ability to infer unknown (but possibly important) quantities such as the galaxies' intrinsic sizes. Similarly, color shifts or image warps may alter the star formation properties or stellar structures of the galaxies, so we don't want to mess with that.

We will center crop the image to \\(144 \times 144\\) pixels using `CropPad()`, which reduces some of the surrounding black space (and other galaxies) near the edges of the images. We will then apply a \\(112 \times 112\\)-pixel `RandomCrop()` for some more translational freedom. This first set of image crop transformations, `item_tfms`, will be performed on images one by one using a CPU. Afterwards, the cropped images (which should all be the same size) will be loaded onto the GPU. At this stage, data augmentation transforms will be performed along with image normalization, which rescales the intensities in each channel so that they have zero mean and unit variance.  The second set of transformations, `batch_tfms`, will be applied one batch at a time on the GPU.


```python
item_tfms=[CropPad(144), RandomCrop(112)]
batch_tfms=aug_transforms(max_zoom=1., flip_vert=True, max_lighting=0., max_warp=0.) + [Normalize]
```

> Note: `Normalize` will pull the batch statistics from your images, and apply it any time you load in new data (see below). Sometimes this can lead to unintended consequences, for example, if you're loading in a test data set which is characterized by different image statistics. In that case, I recommend saving your batch statistics and then using them later, e.g., `Normalize.from_stats(*image_statistics)`.

### Putting it all together and loading the data

We've now gone through each of the steps, but we haven't yet loaded the data! `ImageDataLoaders` has a class method called `from_dblock()` that loads everything in quite nicely if we give it a data source. We can pass along the `DataBlock` object that we've constructed, the DataFrame `df`, the file path `ROOT`, and a batch size. We've set the batch size `bs=128` because that fits on the GPU, and it ensures speedy training, but I've found that values between 32 and 128 often work well.


```python
dls = ImageDataLoaders.from_dblock(dblock, df, path=ROOT, bs=128)
```

Once this is functional, we can view our data set! As we can see, the images have been randomly cropped such that the galaxies are not always in the center of the image. Also, much of the surrounding space has been cropped out.


```python
dls.show_batch(nrows=2, ncols=4)
```

![Batch of galaxies in training data set]({{ site.baseurl }}/images/blog/2020-05-26-training-a-deep-cnn_29_0.png)    


Pardon the excessive number of significant figures. We can fix this up by creating custom classes extending `Transform` and `ShowTitle`, but this is beyond the scope of the current project. Maybe I'll come back to this in a future post!

## Neural network architecture and optimization

![A residual block, the basis for super-deep resnets. Figure from He et al. 2015.]({{ site.baseurl }}/images/blog/resblock.png)

There's no way that I can describe all of the tweaks and improvements that machine learning researchers have made in the past couple of years, but I'd like to highlight a few that really help out our cause. We need to use some kind of residual CNNs (or resnets), introduced by [Kaiming He et al. (2015)](https://arxiv.org/abs/1512.03385). Resnets outperform previous CNNs such as the AlexNet or VGG architectures because they can leverage gains from "going deeper" (i.e., by extending the resnets with additional layers). The paper is quite readable and interesting, and there are plenty of other works explaining why resnets are so successful (e.g., a [blog post by Anand Saha](http://teleported.in/posts/decoding-resnet-architecture/) and [a deep dive into residual blocks by He et al.](https://arxiv.org/abs/1603.05027)).

![One reason why resnets are so much more successful than traditional CNNs is because their loss landscapes are much smoother, and thus easier to optimize. We can also re-shape the loss landscape through our choice of activation function, which we will see below. Figure from Hao Li et al. 2017.]({{ site.baseurl }}/images/blog/resnet_loss.png)

In `fastai`, we can instantiate a 34-layer *enhanced* resnet model by using `model = xresnet34()`. We could have created a 18-layer model with `model = xresnet18()`, or even defined our own custom 9-layer resnet using 
```python
xresnet9 = XResNet(ResBlock, expansion=1, layers=[1, 1, 1, 1]) 
model = xresnet9()
```

But first, we need to set the number of outputs.  By default, these CNNs are suited for the ImageNet classification challenge, and so there are `1000` outputs. Since we're performing single-variable regression, the number of outputs (`n_out`) should be `1`. Our `DataLoaders` class, `dls`, already knows this and has stored the value `1` in `dls.c`.

Okay, let's make our model for real:


```python
model = xresnet34(n_out=dls.c, sa=True, act_cls=MishCuda)
```

So why did I say that we're using an "enhanced" resnet -- an "xresnet"? And what does `sa=True` and `act_cls=MishCuda` mean? I'll describe these tweaks below. 

### A more powerful resnet

The ["bag of tricks" paper](https://arxiv.org/abs/1812.01187) by Tong He et al. (2018) summarizes many small tweaks that can be combined to dramatically improve the performance of a CNN. They describe several updates to the resnet model architecture in Section 4 of their paper. The fastai library takes these into account, and also implements a few other tweaks, in order to increase performance and speed. I've listed some of them below:
- The CNN stem (first few layers) is updated using efficient \\(3 \times 3\\) convolutions rather than a single expensive layer of \\(7\times 7\\) convolutions.
- Residual blocks are changed so that \\(1 \times 1\\) convolutions don't skip over useful information. This is done by altering the order of convolution strides in one path of the downsampling block, and adding a pooling layer in the other path (see Figure 2 of He et al. 2018).
- The model concatenates the outputs of both AveragePool and MaxPool layers (using `AdaptiveConcatPool2d`) rather than using just one.

Some of these tweaks are described in greater detail in [Chapter 14](https://github.com/fastai/fastbook/blob/master/14_resnet.ipynb) of the fastai book, "Deep Learning for Coders with fastai and Pytorch" (which can be also be purchased on [Amazon](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)).

### Self-attention layers

The concept of attention has gotten a lot of, well, *attention* in deep learning, particularly in natural language processing (NLP). This is because the attention mechanism is a core part of the [Transformer architecture](https://arxiv.org/abs/1706.03762), which has revolutionized our ability to learn from text data. I won't cover the Transformer architecture or NLP in this post, since it's way out of scope, but suffice it to say that lots of deep learning folks are interested in this idea.

![An example of the attention mechanism using query f, key g, and value h, to encode interactions across a convolutional feature map. Figure from Han Zhang et al. 2018.]({{ site.baseurl }}/images/blog/self_attention.png)

The attention mechanism allows a neural network layer to encode interactions from inputs on scales larger than the size of a typical convolutional filter. Self-attention is simply when these relationships, encoded via a *query/key/value* system, are applied using the same input. As a concrete example, self-attention added to CNNs in our scenario -- estimating metallicity from galaxy images -- may allow the network to learn morphological features that often require long-range dependencies, such as the orientation and position angle of a galaxy.

In fastai, we can set `sa=True` when initializing a CNN in order to get the [self-attention layers](https://github.com/fastai/fastai/blob/master/fastai/layers.py#L288)!

Another way to let a CNN process global information is to use [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507), which are also [included in fastai](https://github.com/fastai/fastai2/blob/44cc025d9e5e2823d6fd033b84245b0be0c5c9df/fastai2/layers.py#L562). Or, one could even [entirely replace convolutions with self-attention](https://arxiv.org/abs/1906.05909). But we're starting to get off-topic...

### The Mish activation function

Typically, the Rectified Linear Unit (ReLU) is the non-linear activation function of choice for nearly all deep learning tasks. It is both cheap to compute and simple to understand: `ReLU(x) = max(0, x)`.

That was all before Diganta Misra introduced us to the [Mish activation function](https://github.com/digantamisra98/Mish) -- as an undergraduate researcher! He also [wrote a paper](https://arxiv.org/abs/1908.08681) and summarizes some of the reasoning behind it in a [forum post](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299). Less Wright, from the fastai community, shows that [it performs extremely well](https://medium.com/@lessw/meet-mish-new-state-of-the-art-ai-activation-function-the-successor-to-relu-846a6d93471f) in several image classification challenges. I've also found that Mish is perfect as a drop-in replacement for ReLU in regression tasks.

The intuition behind the Mish activation function's success is similar to the reason why resnets perform so well: the loss landscape becomes smoother and thereby easier to explore. ReLU is non-differentiable at the origin, causing steep spikes in the loss. Mish resembles another activation function, [GELU (or SiLU)](https://openreview.net/pdf?id=Bk0MRI5lg), in that neither it nor its derivative is monotonic; this seems to lead to more complex and nuanced behavior during training. However, it's not clear (from a theoretical perspective) why such activation functions empirically perform so well.

Although Mish is a little bit slower than ReLU, a [CUDA implementation](https://github.com/thomasbrandon/mish-cuda/) helps speed things up a bit. We need to `pip install` it and then import it with `from mish_cuda import MishCuda`. Then, we can substitute it into the model when initializing our CNN using `act_cls=MishCuda`.

### RMSE loss

Next we want to select a loss function. The mean squared error (MSE) is suitable for training the network, but we can more easily interpret the root mean squared error (RMSE). We need to create a function to compute the RMSE loss between predictions `p` and true metalllicity values `y`.

(Note that we use `.view(-1)` to flatten our Pytorch `Tensor` objects since we're only predicting a single variable.)


```python
def root_mean_squared_error(p, y): 
    return torch.sqrt(F.mse_loss(p.view(-1), y.view(-1)))

```

### Ranger: a combined RAdam + LookAhead optimzation function

Around mid-2019, we saw two new papers regarding the stability of training neural networks: [LookAhead](https://arxiv.org/abs/1907.08610) and [Rectified Adam (RAdam)](https://arxiv.org/abs/1908.03265). Both papers feature novel optimizers that address the problem of excess variance during training. LookAhead mitigates the variance problem by scouting a few steps ahead, and then choosing how to optimally update the model's parameters. RAdam adds a term while computing the adaptive learning rate in order to address training instabilities (see, e.g., the original [Adam optimizer](https://arxiv.org/abs/1412.6980)).

Less Wright quickly realized that these two optimizers [could be combined](https://forums.fast.ai/t/meet-ranger-radam-lookahead-optimizer/52886). His [`ranger` optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) is the product of these two papers (and now also includes a new tweak, [gradient centralization](https://arxiv.org/abs/2004.01461v2), by default). I have found `ranger` to give excellent results using empirical tests.

So, now we'll put everything together in a fastai `Learner` object:


```python
learn = Learner(
    dls, 
    model,
    opt_func=ranger, 
    loss_func=root_mean_squared_error
)
```

### Selecting a learning rate

Fastai offers a nice feature for determining an optimal learning rate, taken from [Leslie Smith (2015)](https://arxiv.org/abs/1506.01186). All we have to do is call `learn.lr_find()`.

The idea is to begin feeding your CNN batches of data, while exponentially increasing learning rates (i.e., step sizes) and monitoring the loss. At some point the loss will bottom out, and then begin to increase and diverge wildly, which is a sign that the learn rate is now too high.

![Example of the impacts of learning rates and step sizes while exploring a loss landscape. Figure from https://www.jeremyjordan.me/nn-learning-rate/]({{ site.baseurl }}/images/blog/learning-rate.png)

Generally, before the loss starts to diverge, the learning rate will be suitable for the loss to steadily decrease. We can generally read an optimal learning rate off the plot -- the suggested learning rate is around \\(0.03\\) (since that is about an order of magnitude below the learning rate at which the loss "bottoms out" and is also where the loss is decreasing most quickly). I tend to choose a slightly lower learning rate (here I'll select \\(0.01\\)), since that seems to work better for my regression problems.


```python
learn.lr_find()
```

SuggestedLRs(lr_min=0.03630780577659607, lr_steep=0.02290867641568184)

![LR finder plot]({{ site.baseurl }}/images/blog/2020-05-26-training-a-deep-cnn_57_2.png)    


### Training the neural network with a "one-cycle" schedule

Finally, now that we've selected a learning rate (\\(0.01\\)), we can train for a few epochs. Remember that an *epoch* is just a run-through using all of our training data (and we send in one batch of 64 images at a time). Sometimes, researchers simply train at a particular learning rate and wait until the results converge, and then lower the learning rate in order for the model to continue learning. This is because the model needs some serious updates toward the beginning of training (given that it has been initialized with random weights), and then needs to start taking smaller steps once its weights are in the right ballpark. However, the learning rate can't be too high in th beginning, or the loss will diverge! Traditionally, researchers will select a safe (i.e., low) learning rate in the beginning, which can take a long time to converge.

Fastai offers a few optimization *schedules*, which involve altering the learning rate over the course of training. The two most promising are called [`fit_flat_cos`](https://dev.fast.ai/callback.schedule#Learner.fit_flat_cos) and [`fit_one_cycle`](https://dev.fast.ai/callback.schedule#Learner.fit_one_cycle) ([see more here](https://arxiv.org/abs/1708.07120)). I've found that `fit_flat_cos` tends to work better for classification tasks, while `fit_one_cycle` tends to work better for regression problems. Either way, the empirical results are fantastic -- especially coupled with the Ranger optimizer and all of the other tweaks we've discussed.


```python
learn.fit_one_cycle(7, 1e-2)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.142891</td>
      <td>0.368407</td>
      <td>01:57</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.165887</td>
      <td>0.195835</td>
      <td>01:56</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.131631</td>
      <td>0.151854</td>
      <td>01:57</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.107358</td>
      <td>0.099996</td>
      <td>01:57</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.096770</td>
      <td>0.088452</td>
      <td>01:56</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.091173</td>
      <td>0.086019</td>
      <td>01:57</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.088538</td>
      <td>0.085721</td>
      <td>01:56</td>
    </tr>
  </tbody>
</table>


Here we train for only seven epochs, which took under 14 minutes of training on a single NVIDIA P100 GPU, and achieve a validation loss of 0.086 dex. In our published paper, we were able to reach a RMSE of 0.085 dex in under 30 minutes of training, but that wasn't from a randomly initialized CNN -- we were using transfer learning then! Here we can accomplish similar results, without pre-training, in only half the time.

We can visualize the training and validation losses. The x-axis shows the number of training iterations (i.e., batches), and the y-axis shows the RMSE loss.


```python
learn.recorder.plot_loss()
plt.ylim(0, 0.4);
```

![Training losses]({{ site.baseurl }}/images/blog/2020-05-26-training-a-deep-cnn_63_0.png)


## Evaluating our results

Finally, we'll perform another round of [data augmentation](#Transformations-and-data-augmentation) on the validation set in order to see if the results improve. This can be done using `learn.tta()`, where TTA stands for test-time augmentation.


```python
preds, trues = learn.tta()
```

Note that we'll want to flatten these `Tensor` objects and convert them to numpy arrays, e.g., `preds = np.array(preds.view(-1))`. At this point, we can plot our results. Everything looks good!

![Plotting predictions]({{ site.baseurl }}/images/blog/2020-05-26-training-a-deep-cnn_69_0.png)    


It appears that we didn't get a lower RMSE using TTA, but that's okay. TTA is usually worth a shot after you've finished training, since evaluating the neural network is relatively quick.

## Summary

In summary, we were able to train a deep convolutional neural network to predict galaxy metallicity from three-color images in under 15 minutes. Our data set contained over 100,000 galaxies, so this was no easy feat! Data augmentation, neural network architecture design, and clever optimization tricks were essential for improving performance. With these tools in hand, we can quickly adapt our methodology to tackle many other kinds of problems!

`fastai` version 2 is a powerful high-level library that extends Pytorch and is easy to use/customize. As of November 2020, the [documentation](https://docs.fast.ai/) is still a bit lacking, but hopefully will continue to mature. One big takeaway is that fastai, which is all about *democratizing AI*, makes deep learning more accessible than ever before.

**Acknowledgments**: I want to thank fastai core development team, [Jeremy Howard](https://twitter.com/jeremyphoward) and [Sylvain Gugger](https://twitter.com/GuggerSylvain), as well as other contributors and invaluable members of the community, including [Less Wright](https://github.com/lessw2020), [Diganta Misra](https://twitter.com/digantamisra1?lang=en), and [Zachary Mueller](https://muellerzr.github.io/). I also want to acknowledge Google for their support via GCP credits for academic research. Finally, I want to give a shout out to [Steven Boada](https://github.com/boada), my original collaborator and co-author on [our paper](https://arxiv.org/abs/1810.12913).

**Last updated**: 2025-04-23


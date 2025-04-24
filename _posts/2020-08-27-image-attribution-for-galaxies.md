---
title: 'Visualizing deep learning with galaxies, part 2'
date: 2020-08-27
permalink: /blog/2020/08/visualizing-cnn-attributions/
header:
  teaser: https://jwuphysics.github.io/images/blog/gradcam.png
tags:
  - galaxies
  - visualization
  - tutorial
  - original-blog
---

In the previous post, we examined the feature space of galaxy morphological features. Now, we will use the Grad-CAM algorithm to visualize the parts of a galaxy image that are most strongly associated with certain classifications. This will allows us to identify exactly which morphological features are correlated with low- and high-metallicity predictions.


## Galaxies, neural networks, and interpretability

Up til this point, we have been interested in predicting a galaxy's elemental abundances from optical-wavelength imaging. Using the `fast.ai` library, we were able to [train a deep CNN and estimate metallicity to incredibly low error](https://jwuphysics.github.io/blog/2020/05/learning-galaxy-metallicity-cnns/) in under 15 minutes. We then used dimensionality reduction techniques to help [visualize the latent structure of CNN activations](https://jwuphysics.github.io/blog/2020/07/visualizing-cnn-features-pca/), and identified how morphological features of galaxies are associated with higher or lower metallicities.

![Estimating metallicites from imaging]({{ site.baseurl }}/images/blog/WB19_fig1.jpg)

In this post, we will look more closely at the **CNN activation maps** to see which parts of the galaxies are associated with predictions of low or high metallicity. This method of interpretation is sometimes referred to as *image attribution*. We investigate galaxy evolution using [interpretable machine learning in my most recent paper](https://arxiv.org/abs/2001.00018).

One key difference between this analysis and the previous ones is that we will definine the CNN task as a *binary classification* problem rather than a *regression* problem. Once we have trained the classifier to distinguish low- and high-metallicity galaxies, we will be able to produce activation maps for both classes, even though the CNN will only predict one of the two. Setting up this classification task is no more difficult than the previous regression problem using the `fastai` DataBlock API.

The [`fastai` v2 library has been officially released](https://www.fast.ai/2020/08/21/fastai2-launch/), so definitely try it out if you haven't yet! I've previously referred to this as the `fastai2` library, but now it can be found in the main repository: https://github.com/fastai/fastai.
{: .notice}


```python
# imports
from fastai.basics import *
from fastai.vision.all import *
from mish_cuda import MishCuda

seed = 256

ROOT = Path('../').resolve()
```

### Binning the galaxies into metallicity classes

First, we need to define the classes using the parent data set. Below, I plot a histogram of the metallicities for the entire galaxy catalog, where we can see that the mean of the distribution is about \\(Z = 8.9\\).


```python
df = pd.read_csv(f'{ROOT}/data/master.csv', dtype={'objID': str}).rename({'oh_p50': 'metallicity'}, axis=1)

plt.figure(figsize=(6,3,), dpi=150)
plt.hist(df.metallicity, bins=25, color='#003f5c')

plt.ylabel('Number')
plt.xlabel(r'\\(Z\equiv 12\\) + log(O/H)');
```

![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_6_0.png)    


There's no obvious way to define metallicity "classes" since the distribution is unimodal and smooth. We can use `pd.cut` to sort low, medium, and high metallicities into bins \\((8.1, 8.7]\\), \\((8.7, 9.1]\\), and \\((9.1, 9.3]\\). 


```python
df['Z'] = pd.cut(
    df.metallicity, 
    bins=[8.1, 8.7, 9.1, 9.3], 
    labels=['low', 'medium', 'high']
)

df.Z.value_counts()
```

    medium    97097
    low       21403
    high      15421
    Name: Z, dtype: int64


The majority are labeled as medium metallicites, but we will dropping things, such that our remaining data comprises two well-separated classes. The remaining low- and high-metallicity galaxies have slightly imbalanced classes, but this imbalance isn't be severe enough to cause any issues. (In more problematic cases, we could try [resampling](https://pytorch.org/docs/stable/data.html) or [weighting](https://github.com/fastai/fastai/blob/13bf45874516a78068c671d0f1e0618a888457ed/fastai/callback/data.py#L38) our DataLoaders, or weighting the [cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).)


```python
df = df[df.Z != 'medium']

df.Z.value_counts()
```

    low       21403
    high      15421
    medium        0
    Name: Z, dtype: int64


## A CNN classification model

### DataBlocks for classification

Now that we have a smaller DataFrame with a column of metallicity categories (`Z`), we can construct the `DataBlock`. There are a few notable differences between this example and [the previous `DataBlock` for regression](https://jwuphysics.github.io/blog/2020/05/learning-galaxy-metallicity-cnns/):
- we use `CategoryBlock` rather than `RegressionBlock` as the second argument to `blocks`
- we supply `ColReader('Z')` for `get_y`
- we have zoomed in on the images and only use the central 96×96 pixels, which will allow us to interpret the activation maps more easily

Afterwards, we populate `ImageDataLoaders` with data using `from_dblock()`.


```python
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('objID', pref=f'{ROOT}/images/', suff='.jpg'),
    get_y=ColReader('Z'),
    splitter=RandomSplitter(0.2, seed=seed),
    item_tfms=[CropPad(96)],
    batch_tfms=aug_transforms(max_zoom=1., flip_vert=True, max_lighting=0., max_warp=0.) + [Normalize],
)

dls = ImageDataLoaders.from_dblock(dblock, df, path=ROOT, bs=64)
```

We can show a few galaxies to get a sense for what these high- and low-metallicity galaxies look like, keeping in mind that many "normal" spiral galaxies with typical metallicities have been excluded.


```python
dls.show_batch(max_n=8, nrows=2)
```

![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_15_0.png)   


### Constructing a simple CNN

Next, we will construct our model. We will use the `fast.ai` [`ConvLayer`](https://docs.fast.ai/layers.html#ConvLayer) class instead of writing out each sequence of 2d convolution, ReLU activation, and batch normalization layers. After the `ConvLayer`s, we pool the activations, flatten them so that they are of shape `(batch_size, 128)`, and pass them through a fully-connected (linear) layer.


```python
model = nn.Sequential(
    ConvLayer(3, 32),
    ConvLayer(32, 64),
    ConvLayer(64, 128),
    nn.AdaptiveAvgPool2d(1),
    Flatten(),
    nn.Linear(128, dls.c)
)
```

That's it! We have a tiny 4-layer (not counting the pooling and flattening operations) neural network! Since there are only two classes, the DataLoaders knows that `dls.c` = 2 (even though there *was* a third class, galaxies with `medium` metallicities, but we've removed all of those examples from the catalog). 

This final linear layer will output two floating point numbers. Although they might take on values outside the interval \\([0, 1]\\), they can be converted into probabilities by using the softmax function, and this is done implicitly as part of the [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), which we will cover below.

### Optimization and metrics

We can create a `fast.ai` `Learner` object just like before. Since we are working on a classification problem, the `Learner` assumes that we want a flattened version of`nn.CrossEntropyLoss`. Thus, the argument to `loss_func` is optional (unlike in the previous the regression problem, where we needed to specify RMSE as the loss function). In this example, we do also have the option of passing in a weighted or [label-smoothing](https://arxiv.org/abs/1906.02629) cross entropy loss function, but it's not necessary here.

Cross entropy loss is great because it's the continuous, differentiable, negative log-likelihood of the class probabilities. On the flip side, it's not obvious how to interpret this loss function; we're more accustomed to seeing the model accuracy or some other metric. Fortunately, we can supply additional metrics to the `Learner` in order to monitor the model's performance. One obvious metric is the accuracy. We can also look at the [area under curve of the receiving operator characteristic](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) and the [\\(F_1\\) score](https://www.wikiwand.com/en/F1_score) (`RocAuc` and `F1Score`, respectively, in `fast.ai`). If we pass
```python
metrics = [accuracy, RocAuc(), F1Score()]
```
to the `Learner` constructor, these metrics will be printed for the validation set after every epoch of training.


```python
learn = Learner(
    dls, 
    model,
    opt_func=ranger, 
    metrics=[accuracy, RocAuc(), F1Score()]
)
```

Cool! Now let's [pick a learning rate (LR)](https://jwuphysics.github.io/blog/2020/05/learning-galaxy-metallicity-cnns/#Selecting-a-learning-rate) and get started. By the way, shallower models tend to work better with higher learning rates. So it shouldn't be a surprise that the LR finder identifies a higher LR than before (where we used a 34-layer xresnet). 



```python
learn.lr_find()
```

    SuggestedLRs(lr_min=0.05248074531555176, lr_steep=1.5848932266235352)


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_22_2.png)  


We can fit using the one-cycle (`fit_one_cycle()`) schedule as we did before. Here I've chosen 5 epochs just to keep it quick.

Often the `fit_flat_cos()` scheduler works well for *classification* problems (and *not regression* problems). It might be worth a shot if you're training a model from scratch — but if you're using transfer learning, then I recommend sticking to `fit_one_cycle()`, since the "warmup phase" seems to be necessary for good results. 
{: .notice}

```python
learn.fit_one_cycle(5, 8e-2)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>roc_auc_score</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.340562</td>
      <td>0.349320</td>
      <td>0.903313</td>
      <td>0.915260</td>
      <td>0.909530</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.268380</td>
      <td>0.587023</td>
      <td>0.828218</td>
      <td>0.800253</td>
      <td>0.868215</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.259624</td>
      <td>0.212257</td>
      <td>0.935361</td>
      <td>0.930546</td>
      <td>0.945098</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.244578</td>
      <td>0.197691</td>
      <td>0.945817</td>
      <td>0.945994</td>
      <td>0.952809</td>
      <td>00:33</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.227754</td>
      <td>0.185524</td>
      <td>0.950842</td>
      <td>0.950158</td>
      <td>0.957412</td>
      <td>00:33</td>
    </tr>
  </tbody>
</table>


In three minutes of training, we can achieve 95% in accuracy, ROC area-under-curve, and \\(F_1\\) score. We can certainly do better (>98% for each of these metrics) if we trained for longer, used a deeper model, or leveraged transfer learning, but this performance is sufficient for revealing further insights. After all, we want to know which *morphological features* are responsible for the low- and high-metallicity predictions. Indeed, shallower neural networks with fewer pooling layers produce activation maps that are easier to interpret! 

Finally, I would be remiss if I didn't mention that `fast.ai` offers a `ClassificationInterpretation` module! It can be used to plot a confusion matrix.


```python
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
plt.xlabel(r'Predicted $Z$', fontsize=12)
plt.ylabel(r'True $Z$', fontsize=12);
```


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_27_1.png)    


`ClassificationInterpretation` can also plot the objects with highest losses, which is helpful for diagnosing what your model got wrong. Not only that, but it also has the Grad-CAM visualization baked in, so that you can visualize *exactly* which parts of the image it has gotten incorrect. But in the next section, we will implement Grad-CAM ourselves using Fastai forward and backwards hooks. If you're unfamiliar with this topic, it could be helpful to refer to the [*Callbacks and Hooks* section of my previous post](https://jwuphysics.github.io/blog/2020/07/visualizing-cnn-features-pca/#Callbacks-and-hooks) before proceeding to the next section.

## Explaining model predictions with Grad-CAM

### Grad-CAM and visual attributions

We now have a CNN model trained to recognize low- and high-metallicity galaxies. If the model is given an input image of a galaxy, we can also see which parts of the image "light up" with activations based on the galaxy features that it has learned. This method is called **class activation mapping** (see Tong et al. 2015). 

We might expect the CNN to rely on different morphological features for recognizing different classes. If these essential features are altered, then the classification might change dramatically. Therefore, we need to look at features for which the gradient (corresponding to a given feature) is large, and this can be accomplished by visualizing the **gradient-weighted class activation map** (Grad-CAM). This work is detailed in ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," by Selvaraju et al. (2016)](https://arxiv.org/abs/1610.02391). 

![Adapted from Figure 1 of Selvaraju et al. 2016, arXiv veresion 3.]({{ site.baseurl }}/images/blog/gradcam-cat_dog.jpg)

### Hooks for storing activations and gradients

Pytorch automatically computes gradients during the backwards pass for each (trainable) layer. However, it doesn't store them, so we need to make use of the `hook` functionality in order to save them on the forward pass (activations) and backward pass (gradients). The essential Pytorch code is shown below (adapted from the [Fastai book](https://github.com/fastai/fastbook/blob/master/18_CAM.ipynb)).


```python
class HookActivation():
    def __init__(self, target_layer):
        """Initialize a Pytorch hook using `hook_activation` function."""

        self.hook = target_layer.register_forward_hook(self.hook_activation) 
        
    def hook_activation(self, target_layer, activ_in, activ_out): 
        """Create a copy of the layer output activations and save 
        in `self.stored`.
        """
        self.stored = activ_out.detach().clone()
        
    def __enter__(self, *args): 
        return self
    
    def __exit__(self, *args): 
        self.hook.remove()

        
class HookGradient():
    def __init__(self, target_layer):
        """Initialize a Pytorch hook using `hook_gradient` function."""
        
        self.hook = target_layer.register_backward_hook(self.hook_gradient)   
        
    def hook_gradient(self, target_layer, gradient_in, gradient_out): 
        """Create a copy of the layer output gradients and save 
        in `self.stored`.
        """
        self.stored = gradient_out[0].detach().clone()
        
    def __enter__(self, *args): 
        return self

    def __exit__(self, *args): 
        self.hook.remove()
```

Note that the two classes are almost the same, and that all of the business logic can be boiled down to:
1. define a hook function (e.g., `hook_gradient`) that captures the relevant output from a model layer
2. register a forward or backward hook using this function
3. define a Python context using `__enter__` and `__exit__` so that we don't waste memory and can easily call the hooks like `with(HookGradient) as hookg: [...]`


We're interested in the final convolutional layer, as the early layers may have extremely vague features that that may not correspond specifically to any one class.


```python
target_layer = learn.model[-4]

learn.model
```

    Sequential(
      (0): ConvLayer(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ConvLayer(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): ConvLayer(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (3): AdaptiveAvgPool2d(output_size=1)
      (4): Flatten(full=False)
      (5): Linear(in_features=128, out_features=2, bias=True)
    )



### A test image

We also need to operate on a single image at a time. (I think we can technically use a mini-batch of images, but then we'll end up with a huge tensor of gradients!) Let's target this nice-looking galaxy.


```python
img = PILImage.create(f'{ROOT}/images/1237665024900858129.jpg')
img.show()
```

![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_39_0.png)    


We can see that the model is incredibly confident that this image is of a high-metallicity galaxy. 


```python
learn.predict(img)
```


    ('high', tensor(0), tensor([1.0000e+00, 7.1970e-19]))



However, `learn.predict()` is doing a lot of stuff under the hood, and we want to attach hooks to the model while it's doing all that. So we'll walk through this example step-by-step.

First, we need to apply all of the `item_tfms` and `batch_tfms` (like cropping the image, normalizing its values, etc) to this test image. We can put this image into a batch and then retrieve it (along with non-existent labels) using `first(dls.test_dls([img]))`.

We use `dls.train.decode()` to process these transforms, and pass it (the first element, and first batch) into a `TensorImage` which can be shown the same was as a `PILImage`.


```python
x, = first(dls.test_dl([img]))

x_img = TensorImage(dls.train.decode((x,))[0][0])
x_img.show()
```

![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_44_0.png)    

Next, we want to generate the Grad-CAM maps. We can produce one for each class, so let's double-check `dls.vocab` to make sure we know the mapping between integers and high or low metallicity classes. It turns out that 0 corresponds to high, 1 corresponds to low. (We also could have figured it out from the output of `learn.predict()` above.)


```python
dls.vocab
```

    (#2) ['high','low']


At this point, we can simply apply the hooks and save the stored values into other variables. 
- During the forward pass, we want to put the model into eval mode and stick the image onto the GPU: `learn.model.eval()(x.cuda())`. We can then save the activation in `act`.
- We then need to do a backwards pass to compute gradients with respect to one of the class labels. If we want gradients with respect to the low-metallicity class, then we would call `output[0, 1].backward()` (note that this 0 references the lone example in the mini-batch). We can store the gradient in `grad`.
- We might also find it helpful to get the class probabilities, which we temporarily saved in `output`. We can get rid of their gradients and store the two values in `p0` and `p1`, the low-z and high-z probabilities (which sum up to one).


```python
# low-metallicity
class_Z = 1

with HookGradient(target_layer) as hookg:
    with HookActivation(target_layer) as hook:
        output = learn.model.eval()(x.cuda())
        act = hook.stored
    output[0, class_Z].backward()
    grad = hookg.stored
    p0, p1 = output.cpu().detach()[0]
```

Finally, computing the Grad-CAM map is super easy! We average the gradients across the spatial axes (leaving only the "feature" axis) and then take the inner product with the activation maps. In the language of mathematics, we are computing 

\\( \sum_{k} \frac{\partial y}{\partial \mathbf{A}^{(k)}_{ij}} \left [ \frac{1}{N_i N_j}\sum_{i,j} \mathbf{A}^{(k)}_{ij} \right ], \\)

for the \\(k\\) feature maps, \\(\mathbf{A}^{(k)}_{i,j}\\), and the target class \\(y\\). Note that the feature maps have shape \\(N_i \times N_j\\), which ends up in the denominator as a constant, but this just gives us an arbitrary scaling factor. Finally, we stop Pytorch from computing any more gradients and pop it off the GPU with `.detach()` and `.cpu()`. We can then plot the map below.


```python
w = grad[0].mean(dim=(1,2), keepdim=True)
gradcam_map = (w * act[0]).sum(0).detach().cpu()
```


```python
fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=100)

x_img.show(ax=ax)
ax.imshow(
    gradcam_map, alpha=0.6, extent=(0, 96, 96, 0),
    interpolation='bicubic', cmap='inferno'
)
ax.set_axis_off()
```


    
![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_51_0.png)
    


Interesting! Looks like it has highlighted the outer regions of the galaxy. Let's also visualize the high-metallicity parts of the image using the same exact code (except, of course, switching `class_Z = 0` to `class_Z = 1`):


```python
class_Z = 0

with HookGradient(target_layer) as hookg:
    with HookActivation(target_layer) as hook:
        output = learn.model.eval()(x.cuda())
        act = hook.stored
    output[0, class_Z].backward()
    grad = hookg.stored
    p0, p1 = output.cpu().detach()[0]
    
w = grad[0].mean(dim=(1,2), keepdim=True)
gradcam_map = (w * act[0]).sum(0).detach().cpu()

fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=100)

x_img.show(ax=ax)
ax.imshow(
    gradcam_map, alpha=0.6, extent=(0, 96, 96, 0),
    interpolation='bicubic', cmap='inferno'
)
ax.set_axis_off()
```


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_53_0.png)  


## Putting it together

Cool, so now we know how this all works! However, we should actually take only the positive contributions of the Grad-CAM map, because activations are passed through a ReLU layer in the CNN. We can do this by calling `torch.clamp()`. Since `matplotlib` `imshow()` rescales the colormap anyway, the result is that we'll see less of the lower-valued (darker) portions of the Grad-CAM map, but the higheest-valued (brighter) parts will not change.

We will shove all this into a function, `plot_gradcam`, which computes the Grad-CAM maps for low and high metallicity labels, organizes the `matplotlib` plotting, and returns the figure, axes, and CNN probabilities.


```python
def plot_gradcam(x, learn, hooked_layer, size=96):
    
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(8.5, 3), dpi=150)

    x_img = TensorImage(dls.train.decode((x,))[0][0])
    
    
    for i, ax in zip([0, 2, 1], axes):

        if i == 0:
            x_img.show(ax=ax)
            ax.set_axis_off()
            continue

        with HookGradient(hooked_layer) as hookg:
            with HookActivation(hooked_layer) as hook:
                output = learn.model.eval()(x.cuda())
                act = hook.stored
            output[0, i-1].backward()
            grad = hookg.stored
            p_high, p_low = output.cpu().detach()[0]

        w = grad[0].mean(dim=(1,2), keepdim=True)
        gradcam_map = (w * act[0]).sum(0).detach().cpu()

        # thresholding to account for ReLU
        gradcam_map = torch.clamp(gradcam_map, min=0) 

        x_img.show(ax=ax)
        ax.imshow(
            gradcam_map, alpha=0.6, extent=(0, size, size,0),
            interpolation='bicubic', cmap='inferno'
        )
        ax.set_axis_off()
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.02)
    
    return (fig, axes, *(np.exp([p_low, p_high]) / np.exp([p_low, p_high]).sum()))
    
```


And now we can plot it! It looks much better now that we've applied the ReLU. I have also added a few extra captions so that we can see the object ID and CNN prediction probabilities.

We can see not only why the CNN (confidently) classified this galaxy as a high-metallicity system, i.e. its bright central region, but also which parts of the image were most compelling for it to be classified as a low-metallicity galaxy, even though it didn't make this prediction! Here, we see that it has highlighted the far-outer blue spiral arms of this galaxy.

```python
text_dict = dict(
    ha='center',
    color='white',
    fontsize=14
)
```

```python
fig, [ax0, ax1, ax2], p_low, p_high = plot_gradcam(x, learn, hooked_layer=target_layer)

ax0.text(0.5, 0.9, f'1237665024900858129', transform=ax0.transAxes, **text_dict)

ax1.text(0.5, 0.9, 'low-metallicity', transform=ax1.transAxes, **text_dict)
ax1.text(0.5, 0.06, f'$p = {p_low:.3f}$', transform=ax1.transAxes, **text_dict)

ax2.text(0.5, 0.9, 'high-metallicity', transform=ax2.transAxes, **text_dict)
ax2.text(0.5, 0.06, f'$p = {p_high:.3f}$', transform=ax2.transAxes, **text_dict);
```


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_58_0.png)    


### A few more examples

Since we've invested this effort into making the `plot_gradcam()` function, let's generate some more pretty pictures. We can grab some random galaxies from the validation set between the redshifts 0.05 < z < 0.08 (i.e., typical galaxy redshifts), and process them using the trained CNN and Grad-CAM.


```python
val_df = dls.valid.items

objids = val_df[(0.05 < val_df.z) & (val_df.z < 0.08)].sample(5, random_state=seed).objID
```


```python
for objid in objids:
    im = f'{ROOT}/images/{objid}.jpg'
    x, = first(dls.test_dl([im]))
    
    fig, [ax0, ax1, ax2], p_low, p_high = plot_gradcam(x, learn, hooked_layer=target_layer)

    ax0.text(0.5, 0.9, f'{objid}', transform=ax0.transAxes, **text_dict)

    ax1.text(0.5, 0.9, 'low-metallicity', transform=ax1.transAxes, **text_dict)
    ax1.text(0.5, 0.06, f'$p = {p_low:.3f}$', transform=ax1.transAxes, **text_dict)

    ax2.text(0.5, 0.9, 'high-metallicity', transform=ax2.transAxes, **text_dict)
    ax2.text(0.5, 0.06, f'$p = {p_high:.3f}$', transform=ax2.transAxes, **text_dict);
    fig.subplots_adjust()
    fig.show()
```


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_61_0.png)  


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_61_1.png)


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_61_2.png)    


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_61_3.png)   


![]({{ site.baseurl }}/images/blog/2020-08-27-image-attribution-for-galaxies_61_4.png)  


## Conclusions

I hope that you've enjoyed this journey through data visualization techniques using `fast.ai`! One of the goals was to convince you that convolutional neural networks *can be* interpretable, and that methods like Grad-CAM are crucial for understanding what a CNN has learned. Since the neural network makes more accurate predictions than any human, we can gain invaluable knowledge by observing what the model focuses on, potentially leading to new insights in astronomy!

If you're interested in some academic discussion of this sort of topic, then I encourage you to check out my most recent paper, ["Connecting optical morphology, environment, and HI mass fraction for low-redshift galaxies using deep learning"](https://arxiv.org/abs/2001.00018), which delves into a closely related topic. In this work, I use pattern recognition classifier combined with a highly optimized CNN regression model to estimate the gas content of galaxies with state-of-the-art results! Grad-CAM makes an appearance in Figure 11, and is even used for visual attribution in *monochromatic* images (see below). The paper has just been accepted to the *Astrophysical Journal* (ApJ), and is currently in press, but you can view the semi-final version on arXiv now!

![]({{ site.baseurl }}/images/blog/Wu2020-Fig11b.jpg "Figure 11 panel b from my recent paper.")

*This post was migrated to my new blog on 2025-04-23*
{: .notice}
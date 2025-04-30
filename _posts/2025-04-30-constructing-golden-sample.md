---
title: 'Constructing a golden sample for evaluations'
date: 2025-04-30
permalink: /blog/2025/04/constructing-golden-sample/
tags:
 - evaluation
 - llms
 - machine-learning
---

To know how well a classification model truly performs, you need a reliable evaluation dataset. This post explains a practical way to create such a high-quality dataset, often called a "golden sample." This method is especially useful when dealing with situations where one class is much rarer than the other (imbalanced data) and your existing labels might not be entirely accurate. Since creating this golden sample involves carefully selecting examples, it might not have the same mix of positive and negative cases as your full dataset. Therefore, this post also describes how to adjust your final evaluation scores to correct for this, giving you an unbiased measure of your model's performance.

In brief, the "recipe" for building a golden sample involves:
- Categorizing the entire dataset based on agreement and disagreement between the *ML model* (\\(M\\)) and *historical human* (\\(H\\)) labels.
- Performing stratified random sampling across these categories within a predefined review budget (\\(\mathcal{N}_{\text{reviewed}}\\)).
- Conducting careful human consensus review on the sampled items to establish their true labels, forming the golden sample \\(G\\).
- Critically, it presents a correction mechanism to account for the stratified sampling bias. By calculating true outcome rates within each reviewed stratum and re-weighting these rates by the original population counts of each \\(M\\) vs. \\(H\\) category, the method allows for the estimation of accurate performance metrics (TP, FN, FP, TN, Precision, Recall, etc.) that reflect the model's performance across the *entire* original dataset.

This post provides a practical guide to *(i)* construct a golden sample given a historical dataset and limited ability to compile high quality labeled data and *(ii)* compute unbiased model metrics when evaluating model predictions against the golden sample.

**Note**: I wrote this post with the assistance of Gemini 2.5 Pro.
{: .notice}

## Introduction

Let's say that we want to build a golden sample of classifications, \\(G\\), for evaluating our classifier model, \\(M\\). We also have access to a historical sample \\(H\\), which has been labeled by humans, but is likely not good enough for a rigorous evaluation. 

One scenario in which this might arise is when we have historically labeled data by hand, but we now want to operationalize a model to perform this work. We want to see how the ML model performs relative to the historical annotations. In that case, the historical dataset **cannot** be (immediately) treated as the "ground truth," as it would imply that the ML model can never do better than the historical human labels.

So we have binary classifications from \\(M\\) and from \\(H\\) over the entire historical dataset, and we'd now like to construct some smaller sample \\(G\\) with careful human review and consensus; therefore, \\(G\\) will become the ground truth.

We'll also assume that the number of positive classes could be smaller than the number of negative classes, leading to an imbalanced data problem. A real example from my work is identifying [JWST](https://science.nasa.gov/mission/webb/) science papers from recent [arXiv](https://arxiv.org/) preprints. Most papers on the arXiv are not about JWST science, so it doesn't make sense (or it'd be very expensive) to construct a golden sample with proportional numbers of positive classes (JWST papers) and negative classes (non-JWST papers).

As you might have noticed, I'll be using the following shorthand for datasets:
- \\(G\\): Labels from the high-quality **golden sample**
- \\(H\\): Labels from the first-pass **historical labels**
- \\(M\\): Predictions made by the **ML classification model**

Another bit of notation: if the model \\(M\\) classifies something as a positive class, and the golden sample \\(G\\) ground truth is the negative class, then I'll write it as "\\(M^+/G^-\\)".

## Compiling the golden sample

I now present a method for creating a golden sample \\(G\\) by using an initial set of model predictions \\(M\\) and historical labels \\(H\\).

1. Run the classifier model over the entire dataset to get classifications for \\(M\\). We already have classifications for \\(H\\).
2. Categorize the *entire* dataset based on \\(M\\) vs. \\(H\\).[^1] Get the counts for each category.
3. Determine our total review budget, \\(\mathcal{N}_{\text{reviewed}}\\), i.e. how many examples we can afford to have carefully reviewed via consensus.
4. Design a *stratified sampling plan* to select \\(\mathcal{N}_{\text{reviewed}}\\) examples: Decide target review counts for each of the four \\(M\\) vs. \\(H\\) categories.
    - Select a statistically meaningful number of examples (e.g., 30 examples) in each quadrant we sample from. *Randomly* sample the required number of examples from *within* each stratum.
    - Note that we are selecting disagreements (\\(M^+/H^-\\), \\(M^-/H^+\\)) as well as agreements (\\(M^+/H^+\\), \\(M^-/H^-\\)) for review. Hopefully the agreements are faster to review.
    - Perform the human *consensus* review on all \\(\mathcal{N}_{\text{reviewed}}\\) selected examples to determine their *true* classifications.
5. Compile the golden sample \\(G\\). This set consists only of the \\(\mathcal{N}_{\text{reviewed}}\\) examples with their consensus labels.
6. Evaluate \\(M\\) and \\(H\\) against \\(G\\). See the Section below for details on how to compute statistics (while correcting for our *stratified sampling plan*).
    - Generate a confusion matrix between \\(M\\) and \\(G\\) by comparing the model predictions \\(M\\) for the subset in \\(G\\) against their consensus labels in \\(G\\).
    - Generate a confusion matrix between \\(H\\) and \\(G\\) by comparing historical labels \\(H\\) for the subset in \\(G\\) against their consensus labels in \\(G\\).
    - Compute evaluation metrics (precision, recall, F1, specificity, etc.) for both \\(M\\) and \\(H\\) based on these confusion matrices.

## Why can't we just report evaluation metrics from this golden sample?

However, we now need to be careful about evaluation metrics computed using this golden sample with disproportionate numbers of positive and negative classes!

To use the example from the introduction: the true ratio of JWST papers to non-JWST papers might be around 1:20, but we may want to create a golden sample with 50 JWST papers and 150 non-JWST papers – only a 1:3 ratio! If, for instance, we want to balance the expected FP and FN counts, then we must account for the fact that the FP rate computed from the golden sample is actually too low by a factor of roughly 6.7!

To say it another way: our full dataset has a certain distribution across the four \\(M\\) vs. \\(H\\) categories, but previously (in step 4 above) we sampled different proportions across these categories, e.g., we may have reviewed many \\(M^+/H^-\\) but only a small fraction of \\(M^-/H^-\\). Our reviewed golden sample \\(G\\) therefore has a *different proportion of items* from each original quadrant compared to the full dataset. Therefore they are *incorrect estimators* of the *true positives (TP), false negatives (FN), false positives (FP), and true negatives (TN)!*

The \\(M\\) vs. \\(H\\) sample that we reviewed can be split into four quadrants, but recall that the human classifications \\(H\\) are *not* the ground truths.

Meanwhile, we can actually write a confusion matrix using \\(G\\) that were selected for review via stratified sampling. But, these are not representative of the full dataset, so we cannot estimate the calibrated error rates using \\(M\\) vs. \\(G\\):

| | |
|---|---|
| \\(M^+/G^+\\) **≠ TP**! | \\(M^-/G^+\\) **≠ FN**! |
| \\(M^+/G^-\\) **≠ FP**! | \\(M^-/G^-\\) **≠ TN**! |

So, how do we get the confusion matrix using *representative* proportions of \\(G\\)?

| | |
|---|---|
| \\(\text{TP}_{\text{est}}\\) | \\(\text{FN}_{\text{est}}\\) |
| \\(\text{FP}_{\text{est}}\\) | \\(\text{TN}_{\text{est}}\\) |

We can apply corrections based on our knowledge of the *total counts* and the rates of \\(G^+\\) or \\(G^-\\) conditioned on the model/historical classifications.

## De-biasing evaluation metrics

I now explain how to compute a *corrected* confusion matrix that is anchored on the high-quality golden sample, and can be extrapolated for the full historical sample. To estimate these corrected metrics for *M* vs. a re-weighted ground truth \\(G\\), we can perform the following steps:

1. **Write down the total counts in each quadrant:** We need the total count of items in the *entire historical dataset* that fall into each of the four \\(M\\) vs. \\(H\\) categories. Let's designate the *sizes* of these four quadrants: 

| | |
|---|---|
| \\( \mathcal{N}_{\text{total}} (M^+/H^+) \\) | \\( \mathcal{N}_{\text{total}} (M^-/H^+) \\) |
| \\( \mathcal{N}_{\text{total}} (M^+/H^-) \\) | \\( \mathcal{N}_{\text{total}} (M^-/H^-) \\) |

and together they should sum to the total number of items in the dataset.

2. **Analyze Performance for each reviewed quadrant:** For the samples we *did* review in \\(G\\), determine the *true* outcome (\\(G^+\\) or \\(G^-\\)) for each item. Then, calculate the performance *within each reviewed quadrant*:
    - Example: Consider the \\(n(M^+/H^-)\\) items we reviewed from the \\(M^+/H^-\\) quadrant. Compute how many were truly \\(G^+\\) and how many were truly \\(G^-\\) after review.
        - \\(n_{\text{reviewed}}(M^+/H^- \rightarrow G^+)\\): Count of reviewed \\(M^+/H^-\\) items that are actually \\(G^+\\).
        - \\(n_{\text{reviewed}}(M^+/H^- \rightarrow G^-)\\): Count of reviewed \\(M^+/H^-\\) items that are actually \\(G^-\\).
    - Calculate the *rate* \\(\mathcal{R}(G^{\pm} \mid \cdot)\\) of actual positives/negatives within this reviewed sample – i.e., conditioned on the original model and historical classifications. For example:
        - \\(\mathcal{R}(G^+ \mid M^+/H^-) = n_{\text{reviewed}}(M^+/H^- \rightarrow G^+) / n(M^+/H^-)\\)
        - \\(\mathcal{R}(G^- \mid M^+/H^-) = n_{\text{reviewed}}(M^+/H^- \rightarrow G^-) / n(M^+/H^-)\\)
    - Repeat this process for all four quadrants sampled from (\\(M^+/H^+\\), \\(M^-/H^-\\), \\(M^+/H^-\\), \\(M^-/H^+\\)), calculating the rates \\(\mathcal{R}(G^+ \mid \text{quadrant})\\) and \\(\mathcal{R}(G^- \mid \text{quadrant})\\). These rates represent the *best estimates* of the *ground truth* probabilities for items that fall into each \\(M\\) vs. \\(H\\) quadrant.

3. **Estimate the confusion matrix for model classifications (\\(M\\) vs. \\(G\\)):** Combine the population sizes and the *within-quadrant rates* to estimate the counts for the overall confusion matrix. Specifically, we can compute:
  - **Estimated True Positives (\\(\text{TP}_{\text{est}}\\)):**
    $$ \text{TP}_{\text{est}} = \mathcal{N}_{\text{total}}(M^+/H^+) \cdot \mathcal{R}(G^+ \mid M^+/H^+) + \mathcal{N}_{\text{total}}(M^+/H^-) \cdot \mathcal{R}(G^+ \mid M^+/H^-). $$
    Take all items \\(M\\) classified as positive. For those also \\(H\\) positive, multiply by the rate they turned out to be truly \\(G^+\\). For those where \\(H\\) was negative, multiply by the rate they turned out to be truly \\(G^+\\).
    - **Estimated False Negatives (\\(\text{FN}_{\text{est}}\\)):**
    $$ \text{FN}_{\text{est}} = \mathcal{N}_{\text{total}}(M^-/H^+) \cdot \mathcal{R}(G^+ \mid M^-/H^+) + \mathcal{N}_{\text{total}}(M^-/H^-) \cdot \mathcal{R}(G^+ \mid M^-/H^-). $$
    Take all items \\(M\\) classified as negative. For those \\(H\\) called positive, multiply by the rate they turned out to be truly \\(G^+\\). For those where \\(H\\) was also negative, multiply by the rate they turned out to be truly \\(G^+\\).
    - **Estimated False Positives (\\(\text{FP}_{\text{est}}\\)):**
    $$ \text{FP}_{\text{est}} = \mathcal{N}_{\text{total}}(M^+/H^+) \cdot \mathcal{R}(G^- \mid M^+/H^+) + \mathcal{N}_{\text{total}}(M^+/H^-) \cdot \mathcal{R}(G^- \mid M^+/H^-). $$
    Take all items \\(M\\) classified as positive. Multiply by the rates they turned out to be truly \\(G^-\\) within their original \\(H\\) classifications.
    - **Estimated True Negatives (\\(\text{TN}_{\text{est}}\\)):**
    $$ \text{TN}_{\text{est}} = \mathcal{N}_{\text{total}}(M^-/H^+) \cdot \mathcal{R}(G^- \mid M^-/H^+) + \mathcal{N}_{\text{total}}(M^-/H^-) \cdot \mathcal{R}(G^- \mid M^-/H^-). $$
    Take all items \\(M\\) classified as negative. Multiply by the rates they turned out to be truly \\(G^-\\) within their original \\(H\\) classifications.

4. **Calculate final corrected metrics:** Use these estimated counts to build our *final confusion matrix* for \\(M\\) vs. \\(G\\), from which we can calculate precision, recall, F1 score, etc. These corrected metrics will now reflect the model's performance on the original, imbalanced data distribution, despite being initially sampled in a stratified manner.

This approach can also be applied to determine the corrected confusion matrix elements for *historical* classifications, \\(H\\) vs. \\(G\\).

## Summary

I offer a practical framework for constructing a golden sample (\\(G\\)) and accurately evaluating classifier model predictions (\\(M\\)) in the common situation where a complete, perfectly labeled ground truth dataset is unavailable or prohibitively expensive to create, especially with class imbalance, but a noisier historical dataset exists (\\(H\\)). By employing stratified sampling based on model (\\(M\\)) and historical (\\(H\\)) classifications, followed by careful human review and an important statistical re-weighting step, this recipe allows us to generate statistically unbiased estimates for confusion matrix components (TP, FN, FP, TN) and derived metrics like precision and recall. The corrected metrics accurately reflect the classifier's expected performance on the full, original data distribution, overcoming the limitations imposed by the targeted sampling strategy. This approach can be also applied to assess the quality of the historical labels against the established golden sample – providing a useful metric of *how good is good enough*. Ultimately, this recipe provides a valuable tool for reliable model monitoring, evaluation, and iterative improvement in real-world 0operations.

However, there are still some caveats. To name a few:
- Datasets can drift over time, which means that historical performance is not indicative of future performance. For example, the number of JWST science papers before the observatory launched (pre-2022) is essentially zero, and the occurence of *bona fide* JWST papers may continue to evolve over time.
- Within each \\(M\\) vs. \\(H\\) quadrant, examples may vary in difficulty or ambiguity. It's likely that there are other factors that influence this. In the example I gave, the length of the paper, or the intended journal for submission might covary with the model performance in a way that is not captured here.
- It is very difficult to achieve perfect human consensus on the golden sample labels! At STScI, we found that multiple rounds of review led to clearer guidelines and definitions, but ultimately there was still some scatter in human labels. This disagreement can be [quantified](https://en.wikipedia.org/wiki/Inter-rater_reliability) but there's no silver bullet to resolving disagreement.

---

[^1]: The \\(M\\) vs. \\(H\\) agreement/disagreement resembles a confusion matrix, but we should not think about it this way. This is because historical classifications \\(H\\) are *not* the ground truths! Instead, we should think of this as a way to stratify our historical dataset into four quadrants for review.
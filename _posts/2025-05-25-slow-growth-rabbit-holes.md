---
title: 'The benefits of slow growth, misguided rabbit holes, and painful mistakes'
date: 2025-05-25
permalink: /blog/2025/05/slow-growth-rabbit-holes/
tags:
  - academia
  - personal
  - productivity
---

I am a self-confessed *productivity junkie*. I hate wasting time. And if you scroll through social media, or even my blog posts, you might think that the typical research or learning process is just a happy, monotonic hill climb, capped off with regular announcements of new discoveries or gained expertise. But what if the most important lessons emerge not from unencumbered progress, but rather from seemingly aimless pursuits and the frustration of doing things badly? This post is a tribute to all those times we got stuck and emerged with nothing to show for it, because those "unproductive" moments lead to some of the most important lessons we can ever learn.

A lot of this post stems from my own experience, and I hope that they're useful for you too. (But one of the takeaways here is that *sometimes you have to make your own mistakes* in order to learn.) Here are a few other blog posts that have impacted my thinking on productivity:
* [Impact, agency, and taste](https://www.benkuhn.net/impact/) by Ben Kuhn
* [How I think about my research process](https://www.alignmentforum.org/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK) by Neel Nanda (note that there are three parts)
* [Some reasons to work on productivity and velocity](https://danluu.com/productivity-velocity/) by Dan Luu
* [Hacker School's Secret Strategy for Being Super Productive (or: Help.)](https://jvns.ca/blog/2014/03/10/help/) by Julia Evans
* [Every productivity thought I've ever had, as concisely as possible](https://guzey.com/productivity/) by Alexey Guzey
* [The top idea in your mind](https://www.paulgraham.com/top.html) by Paul Graham

I'm sure there are more that I've internalized, but can't quite remember right now; feel free to reach out to me if you other interesting ones.


## The exploratory phase

It's hard to quantify the value in trying out new research directions. Obviously you can't wander aimlessly *all the time*, or spend all your free time listening to [NotebookLM](https://notebooklm.google.com/) audio overviews of random papers that piqued your interest. But many of my best ideas emerged despite me not beginning with a tangible goal.

Back when I was in grad school,[^1] I noticed that a [Kaggle Galaxy Zoo challenge](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge) was solved using deep convolutional neural networks (CNNs). I was very interested in applying deep learning to galaxies, so it was gratifying to see [Sander Dieleman](https://sander.ai/) et al. accurately [predict citizen scientist vote fractions of galaxy morphology]([published](https://arxiv.org/abs/1503.07077))  purely using image cutouts. 

Motivated by this successful application... I decided to proceed by bashing every project I could find with this newfound hammer. After all, I was a curious grad student wielding a powerful method. What else did you expect? Classifying galaxy morphology had already been done before, but I recognized that you could predict all sorts of other galaxy properties, e.g., whether galaxies were merging, separating compact galaxies from stars, predicting the bulge-to-disk ratio, etc.

Along the way, though, I noticed that nearly everyone was interested in **classification** problems, e.g., identifying galaxy morphological type or galaxy mergers, but these seemed to be an incredibly limited class of problems. After all, the cosmos is *weakly modal*, and although astronomers loved to classify things, these categories are honestly quite arbitrary.[^2] I was far more interested in **regression** problems, e.g., how does the galaxy's star formation rate or chemical abundance scale with its appearance? Up until ~2017, very few people had addressed the idea of *regression* deep learning problems in astronomy.

Anyway, after a few months of going down random rabbit holes, I realized that there were loads of interesting regression problems that hadn't been addressed with deep learning. I chatted with [Stephen Boada](https://www.linkedin.com/in/theboada/), and later on consulted with [Eric Gawiser](https://www.physics.rutgers.edu/~gawiser/) about these ideas; we quickly honed in on the task of predicting galaxy metallicity from images. You can read more about that [here](https://jwuphysics.github.io/blog/2020/05/exploring-galaxies-with-deep-learning/).

These exploratory phases are helpful for letting your mind make free-form connections; diving down rabbit holes is basically feeding that part of your brain. But watch out for the slippery slope: it's tempting to put out theories without ever figuring out how to evaluate (or invalidate) them. In other words, it's fine to follow random streams of consciousness, but eventually you'll need to land on a *well-posed* research question. Otherwise, you'd never crystallize any kind of problem worth solving! 

I think about this transition as going from the *exploratory/meandering* phase to *mapmaking* phase. That transition happens once you have a falsifiable hypothesis, after which you can begin charting out a plan to implement those tests. Let's talk about the *mapmaking* phase.

## From meandering to mapmaking: distilling down to a one-sentence hypothesis

One of the most important lessons I've learned is this: **Whenever you are in an exploratory phase, look for every opportunity to distill your ideas into a testable, one-sentence hypothesis.**

Side note: LLMs are extremely helpful here! As described in a [previous post](https://jwuphysics.github.io/blog/2025/04/four-ways-i-use-llms/), under the heading *1. Exploring ideas and surveying prior art*, I lean on LLMs to (i) critique my vague thoughts, (ii) decompose promising ideas into atomic concepts, and (iii) survey the literature to see whether these ideas have been implemented before. If you're interested in critiquing your thoughts, then you must avoid [LLM sycophancy](https://openai.com/index/sycophancy-in-gpt-4o/) at all costs! Try a prompt based on something like this:

> Please critique my thoughts on *Topic_X* (appended below). Is my hypothesis vague or incomplete? How might it be tested? Has it been done before? Include diverse opinions or parallel ideas from the literature on *Topic_X* or related concepts.
>
> {{topic_X}}

If you can't (eventually) articulate a testable hypothesis, then you should be slightly worried. Either you are still learning the ropes for a new topic (*good!*), or you are familiar with a topic/method but cannot figure out what you want to do with it (*not good!*). Give yourself a hard deadline (e.g. a week) to distill a one-sentence hypothesis from all the information you've gained while chasing down rabbit holes, and if you still can't come up with anything concrete, then put those thoughts on the backburner.

As soon as you come across a new idea, rigorously consider the following:
* Do I understand the method well enough to sketch out the pseudo code?
* Do I understand the prior art and potential, e.g., is it sufficiently novel and/or impactful?
* Can I write down a testable hypothesis?

If you can't address these questions, then put the idea on the backburner. In fact, for more experienced researchers, you'll want a much tighter feedback loop; each of these questions should be answerable within a few minutes. I come up with a dozen nebulous ideas every day, so it's imperative that I set a five minute deadline for constructing a hypothesis, and if it fails to meet that bar, then I let it sink back into my subconscious. 

Alternatively, there are cases in which it's better to have a **strong** rather than a quick feedback loop. I'll touch on that in the next section.

But once you **do** find a testable hypothesis, then try to write it down into a single sentence. This can be tricky, but the point of the exercise is to practice conveying the essence of the hypothesis, and winnowing out extraneous details. Once you have something specific enough to actually disprove, and you're satisfied that it captures the core research question you'd like to solve, then congratulations! You're done exploring (for now) — it's time for mapmaking.

Here's a concrete example of when I failed. Around 2020, I got interested in generative models for galaxies. "I want to apply VAEs/GANs/diffusion models to astronomy" sounds great when you're reading the DDPM paper, but it's also a completely vague and unfalsifiable thought — there's no scientific question in here. You could spend months on that without it amounting to anything. But instead, we could start thinking about more testable hypotheses:
* Can generative models construct galaxy images at high enough fidelity to forecast new survey observations? (*This is still too vague, but we're getting closer.*)
* Can generative models predict JWST-like mid-infrared images from HST imaging to a reduced chi-squared value of about 1, thereby confirming a [tight connection between galaxies' optical and infrared light](https://arxiv.org/abs/2503.03816)?
* If we generate mock survey images based on $N$-body simulations with realistic galaxy morphologies and brightnesses, and select galaxies via aperture photometry based on rest-frame optical colors, then does it result in a biased matter power spectrum relative to doing the same with halo occupation distribution models that *don't* include morphology?

I'm not saying these are particularly good research ideas. But the second and third are definitely more testable than the first one, and each of those three are far more useful than "can we train VAEs/GANs/diffusion models over my favorite galaxy dataset?" 

Specifying the hypothesis in this way makes it obvious that the hypothesis could be true or false. Better yet, it implies *how* the hypothesis might be validated or falsified. Still, we might have a vague but interesting idea, and we can think of multiple tests that could (in)validate parts of this unclear idea. In that case, we can hierarchically break down the target idea (e.g., *the latent space of generative models trained on galaxy images has a bijective map to the space of astrophysical properties*) into more specific hypotheses, like:
* A generative model trained on SDSS galaxy image cutouts will have some linear combination of latent vectors that have high Pearson correlation coefficient with HI velocity width.
* A generative model's latent vector that is positively correlated with inclination angle will also be anticorrelated with star formation rate from optical tracers.

I want to admit that I've often gotten stuck with a tantalizing idea, but couldn't (or didn't) find a way to test a concrete hypothesis. For example, I wanted to make "superresolution" work in astronomy, and even wrote some [blog](https://jwuphysics.github.io/blog/2021/01/galaxy-unets/) [posts](https://jwuphysics.github.io/blog/2021/12/galaxy-gans/) about it. Whereas smarter researchers like Francois Lanusse came up with [genuinely useful applications for generative modeling](https://arxiv.org/abs/2008.03833), I was just messing around. **I hadn't ever left the exploratory phase, and although I believed I was mapping out a trajectory, I had no destination in mind!**

## The irreplaceable experience of doing a really, really bad job

Let's actually zoom out for a moment, because there's an important lesson to be learned here.

I mentioned that it's useful to have tight or quick feedback loops: arrive at testable hypotheses so that you can begin the real work of validating or falsifying them. This is the actual learning process. Aimless exploration often has the *appearance* of learning, but carries little substance... except for one critical meta-lesson. **Sometimes you simply need to experience the feeling of aimlessness in order to learn how to overcome it.**

Perhaps you don't know what kind of statistical methods are needed to confirm or falsify a hypothesis. Or maybe you need to collect a lot more data, and that'll take a year. Or perhaps your PhD supervisor really thinks that the conjecture is true, but you can't figure out how to confirm it. In all these cases, you're left without any useful feedback, or an obvious way to proceed, so you just flounder about and watch weeks, months, maybe even years go by. **You're absolutely failing at your task. Keep it up!** 

I can give a personal anecdote: I got totally stuck during my first research project in graduate school. And by stuck, I mean *I was absolutely going down random wrong rabbit holes and wasting my time*. I was tasked with "stacking far-infrared, dust continuum, and gas spectral lines from high-redshift cluster galaxies." Stacking just means that, since we know the locations (and recession velocities) of a sample of galaxies, we can *average* how bright those galaxies are using some other measurements. Simple enough, right? But at the time, I didn't know what I didn't know:
* At far-infrared wavelengths, there is a considerable background emission from high-redshift galaxies, and especially so in galaxy cluster fields where gravitational lensing can amplify the background flux. 
* Far-infrared emission detected via bolometers generally has very low spatial resolution, so that if you are stacking lots of galaxies within a small angular size, then you'll accidentally double count sources (that are well-separated at optical wavelengths, but completely "blended" or overlapping at long wavelengths).
* I was relying on a sample of galaxy clusters spanning redshifts between $0.3 < z < 1.1$, meaning that my *flux-limited samples* would result in completely heterogeneous constraints spanning nearly two orders of magnitude in luminosity or mass.
* Ultimately, stacking would not have detected anything in the highest-redshift clusters unless the *average cluster member* had lots of dust-obscured star formation — a very bold assumption.
* There were a few galaxies that were actually detected in long-wavelength emission, but there were so few that it was impossible to make any kind of statistical statement about them.
* The whole project was extremely challenging, and probably could have been framed as "we expected to find nothing in these extremely massive galaxy clusters, and yet we found some rare dusty, gas-rich, star-forming galaxies!" (A member of my qualifying exam committee actually said this, but I didn't take it to heart.)

To be clear, my PhD advisor was extremely supportive and helpful in providing high-level guidance. I just happened to be pushing in a challenging research direction, and I was too inexperienced in astronomy to have salient intuitions on how to interpret my (null) results. After navigating these unexpected twists and turns over three years,[^3] I finally had some results and a paper draft to circulate among my co-authors.

One of the co-authors (I won't say who) remarked, "*Whew. That was a slog to read through. It really reads like a student's first paper... you gotta make it more interesting than that.*"

I had dutifully reported the results from every misguided rabbit hole, mentioned our incorrect motivations, and carefully explained the statistical procedures that were far too weak to result in anything. But my co-author's message cut through all the nonsense: **Give your audience a reason to read this. Nobody cares about your hard work.**

## Don't skip the struggle, don't repeat the struggle

This was the lesson I needed to learn, and I'm not sure there were any shortcuts. Instead of just learning what *to do*, I had to suffer through and internalize what *not to do*. I was embarrassed at how inefficient I was and completely sick of this project.[^4] But I had to see it through. 

Ultimately, it took over **four years** to get my [**first paper**](https://arxiv.org/abs/1712.04540) published. The end result wasn't great; in hindsight, I could have written it much better, and I probably could have made it sound far more interesting. 

And yet, I think this was one of the best things to happen to me as a graduate student. **I truly believe that there's no substitute for the experience of struggling to conclude a project, and then pushing through anyway**. Each and every fruitless endeavor builds intuition for future projects. Critiques and criticisms hone your research taste. Slow learning by trial and error ensures that you have an airtight understanding of those topics.

The greatest part of this story is that I've already gone through this journey once, so I don't need to repeat it! I'm sure that I'll run into new obstacles and challenges, and that I'll be frustrated with my lack of productivity, but — with the benefit of hindsight — I can appreciate the slow learning process for what it is. 

---

[^1]: I've previously written about my journey, including my serendipitous grad school years, [here](https://jwuphysics.github.io/blog/2024/01/two-years-in-the-tenure-track).
[^2]: There's a whole blog post in here, but in essence: many astrophysical phenomena exist along a continuum. That continuum *might* be bimodal, like elliptical vs disk galaxies, or broad-line vs narrow-line active galactic nuclei, or Type O/B/A/etc stars, but there is rarely a firm separation of classes. Sure, there are obvious class distinctions like star–galaxy separation, but you know I'm not talking about that. If you want to hear more, stay tuned for a future post, or check out this [MLClub debate](https://youtu.be/kpMXCcGyydU&t=1529) back in 2021.
[^3]: I pulled *so many* all-nighters during my first year of graduate school. Probably chasing down loads of random rabbit holes on that first project. And for what? None of it was useful for writing my first paper. And yet it *was* useful, because now I know that I don't need to repeat the experience!
[^4]: [Josh Peek](https://jegpeek.space/) (and I'm sure there's earlier attribution) often says, "hating your paper is a necessary but not sufficient condition to getting it published."
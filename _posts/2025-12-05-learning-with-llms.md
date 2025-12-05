---
title: 'Learning with LLMs'
date: 2025-12-05
permalink: /blog/2025/12/learning-with-llms/
tags:
  - education
  - llms
  - personal
---

AI is here, and its impacts on education cannot be overstated. Let's put aside the issues of cheating; I assume that you *want* to learn, perhaps with the assistance of LLMs *if* they are actually helpful. But how can you be sure that you're not using AI as a crutch (e.g., outsourcing your thinking), versus using it to augment learning (e.g., revealing gaps in understanding, bypassing blockers that prevent learning, or tailoring education to your style)? In this post, I provide an analogy between learning and phase transitions in statistical mechanics, and describe recommendations and warnings on using LLMs in different learning scenarios.

## Introduction

Even though it *feels* like you're learning when you talk to LLMs, you might be selling yourself short: you're liable to self-deception, thinking that you have a strong grasp of a topic when you do not. Nevertheless, I also believe that there are certain scenarios where AI *can* be used without compromising your ability to think. Exactly where the distinction occurs can vary from person to person, because everyone has different learning styles.

In my view, there are three regimes of learning: we begin by *building familiarity and intuition*, progress toward *spotting patterns and making connections*, and finally achieve *robust knowledge retention*. I'm going to use the idea of a phase transition from statistical mechanics as a high-level analogy for learning; these three regimes are really two phases, a subcritical phase (building intuition) and supercritical phase (robust knowledge), separated by a phase transition. It's in this phase transition — the critical point — where connectivity between learned ideas becomes possible.[^1]

Throughout this post, I'm going to use two examples of learning: *addition* and *the covariance matrix*. The first one is based on me teaching my five-year-old daughter simple arithmetic, and the second reflects my own learning journey from an undergraduate physics major to a tenure-track researcher. In the example of learning about single-digit addition, you can imagine "summarization via an LLM" is like "asking dad for the answer" or "asking dad *why*". In the second example, you can just imagine asking ChatGPT for intuitions about *covariance* or *PCA*. 

## Subcriticality: The Early Phase of Learning

At the earliest stage of learning, concepts don't really leave a lasting impression. A number like *5* is just a symbol, and a word like *covariance* is simply some letters on a page. It is painfully difficult to connect these concepts to other ideas that you're supposed to be learning at the same time, e.g., the number *4* is smaller and the number *6* is larger than *5*. Or maybe that your sample covariance matrix can be *diagonalized* to find *eigenvectors* and *eigenvalues*. And you could maybe remember these facts and procedures. But if somebody described Principal Components Analysis using some other choice of words, then you'd have no idea that they were describing the same ideas!

The problem is that in this **subcritical** phase, concepts easily fizzle out. They're totally disconnected from your intuition, because that intuition needs to be formed via related concepts. If you have no intuition for all of the properties of the number *5* — that it is odd, that it is half of ten, that it is three less than eight, etc., then it might as well just be any random symbol on a page. You see symbols, but not the underlying *structure* of these numbers, probably because you simply haven't spent enough time staring at them and thinking about their relationships.

At this stage, it might be easiest to just learn via rote memorization. (This varies by person — I have horrible memory, so I hate this phase of learning.) Back in undergrad, I remember buying the PDF for *[Learn Python the Hard Way](https://learnpythonthehardway.org/)*, a textbook for learning the Python programming language. I printed out each of the pages, so that I would have to manually type in code, rather than copy-paste it! This helped me build up muscle memory and think about the Python code as I typed it in.

Lots of folks have found that spaced repetition is the best way to improve learning at this stage (e.g., [Gwern](https://gwern.net/spaced-repetition), [Anki cards](https://apps.ankiweb.net/), [HN discussions](https://news.ycombinator.com/item?id=44020591)). At its core, spaced repetition is just testing how well you've remembered things over progressively longer time periods — the intervals get shorter if you forget something, and get longer if you are able to continue recalling it.

While there's certainly some benefit to *assembling* the spaced repetition system (i.e., constructing a deck of Anki cards), I think that the most valuable technique is to regularly *use* it. That is, it's okay if somebody else procures the answer for you. It's okay if the definition of *sample covariance matrix* comes from a textbook or Wikipedia or Claude. You're still at the subcritical phase, and you'll need more exposure to the concepts before things start to click.

However, this phase of learning is still meant to be somewhat difficult. You should expect friction! Recall that it's easy to nod your head in agreement when you read something that makes sense, but it's far more difficult — and valuable — to re-derive it yourself.[^2] Once you become somewhat familiar enough with a concept, then it becomes much more rewarding to test that knowledge to see if you've reached the critical point.

I think that current LLMs are highly useful for testing your knowledge. For example, to assist my own learning, I frequently use a "Socratic" prompt with Gemini (although it's out of date, and note that it was written with the assistance of Gemini 2.5 Pro):
> Purpose and Goals:
> * Help users refine their understanding of a chosen topic (Topic X).
> * Facilitate learning through a Socratic method, prompting users to explain concepts and asking probing questions.
> * Identify and address misunderstandings by testing the user's conceptual knowledge.
> * Focus on sharpening the user's intuition and conceptual understanding rather than rote memorization.
> [...]
The entire prompt can be found [here](https://gist.github.com/jwuphysics/dace4952279e4b047c6c0591d09895d6).

This method of using LLMs should not make you reliant on AI. It does not outsource your thinking. Like the [*How To Solve It* method](https://en.wikipedia.org/wiki/How_to_Solve_It) devised by the great mathematician George Pólya, and the eponymous [SolveIt platform](https://solve.it.com/) run by great educator Jeremy Howard, my aim here is to demonstrate how to use LLMs as a *personalized tool* for testing your understanding. LLMs are now powerful enough that (for most topics), they can spot holes in your thinking; however, given their tendencies toward sycophancy, LLMs must be prompted carefully.

## Supercriticality: The Late Phase of Learning

At some point, the dots really start to connect. Beyond the critical point, all your knowledge is linked together. You have intuitions for concepts that you may not have heard of. You're so comfortable with addition, that you also intuitively grasp concepts like associativity (*1 + 3 = 3 + 1*) or inverses (*adding one to three makes four, so taking away one from four makes three*), even though you may not have heard of (or recall) the jargon from algebra or group theory. In any event, you have a robust conceptual understanding, and all that remains is to give names to these well-understood concepts. 

In this phase, learning *should* feel easy and fun. There are likely still gaps in your knowledge, but it's quite straightforward to fill them in. Your knowledge is robust even when you're missing certain pieces of information because you've trodden all around that *terra incognita*, so new knowledge doesn't dramatically upend your understanding. 

My late father had a PhD in chemistry. He loved to personify everything and attach *feelings* to them: oxygen *wants* to gain two electrons, the molybdenum *likes* to react when this catalyst is present, etc. We develop a similar feel for concepts when our understanding passes the critical point. And this intuition is vital for pursuing novel research ideas and making scientific discoveries.

Or, you can plausibly extend your knowledge to other domains because you have crystallized the relevant intuitions. For example, in the excellent pedagogical text [*Data analysis recipes: Fitting a model to data*](https://arxiv.org/pdf/1008.4686), David Hogg et al. write:
> The inverse covariance matrix appears in the construction of the \(\chi^2\) objective function like a linear "metric" for the data space: It is used to turn a N-dimensional vector displacement into a scalar squared distance between the observed values and the values predicted by the model. This distance can then be minimized. This idea—that the covariance matrix is a metric—is sometimes useful for thinking about statistics as a physicist.

While physicists can be guilty of thinking that they can leap into other fields ([relevant XKCD](https://xkcd.com/793/)), they often *do* have a strong grasp of mathematics and physical intuition. This combination is invaluable at the supercritical stage: the language of mathematics often translates well to other disciplines, while the intuition from physics can be helpful for predicting dynamics or phenomena given those mathematical laws. 

In the supercritical phase of learning, LLMs can be helpful. They are pretty good at identifying analogous concepts in alternate fields that you might not know about, acting as both the proverbial *water cooler* and the multidisciplinary scientists that congregate around it. LLMs can also be used to quickly refresh ideas that you have briefly forgotten, like going back to reference your old textbooks to check some relevant information. However, this can also be dangerous if you *think* you're past the critical point — but in reality you aren't (often, because your confidence is inflated by talking too much to LLMs).

## The pitfalls of using LLMs to help you learn

I'm reminded of one salient point from [Neel Nanda's delightful essays on research taste](https://www.alignmentforum.org/posts/hjMy4ZxS5ogA9cTYK/how-i-think-about-my-research-process-explore-understand). While not the main focus of those pieces, he explains (emphasis mine):
> Junior researchers often get stuck in the early stages of a project and don’t know what to do next. In my opinion this is because they think they are in the **understanding stage**, but are actually in the **exploration stage**.

In other words, junior researchers can sometimes believe that they have crystallized their understanding of a topic (i.e. supercritical), when in reality they are still in an earlier stage of learning (subcritical)! This is particularly worrisome when LLMs can summarize topics in easily understandable ways, deceiving junior researchers into feeling like they confidently understand a topic because they've understood the simple LLM summarization.

LLMs are truly a double-edged sword for learning. 

On one hand, they can be helpful by testing your knowledge in new ways (e.g. the Socratic approach I mentioned above; I also have another prompt that quizzes me with PhD qualifying exam-like questions). LLMs can help you get unstuck when your canonical text doesn't explain something in a manner intelligible to you. They can get rid of irrelevant roadblocks (e.g., you're learning about neural network optimization, but stuck in CUDA dependencies hell). LLMs can spin up highly individualized games that help you learn concepts in a way that's much more fun than doing practice problems.[^3] 

On the other hand, LLMs can leave you with a completely shallow understanding of a topic — while you feel like you totally understand it all! This is compounded by the fact that LLMs will tend toward positivity. Do not let your confidence be bolstered by hollow AI validation. Be vigilant and skeptical, because uncritical use of AI tools will absolutely inhibit your learning. 

## The pitfalls of using LLMs for summarization

One can imagine a world where AI widens the gap between those who [practice writing and those who do not](https://paulgraham.com/writes.html). This is problematic because — as all experienced researchers know — [writing is thinking](https://www.nature.com/articles/s44222-025-00323-4). If we don't practice writing, then we shut the door on an entire mode of thinking. 

*But what about reading?* Sometimes it feels like a slog to read through long articles, especially technical, information-dense academic papers. Why not just get it all distilled into a single paragraph? Or turn it into a podcast-like audio overview? As I [wrote on social media](https://bsky.app/profile/jwuphysics.bsky.social/post/3m65eelo5gs2f), *using AI to summarize information is also a way to outsource your thinking*. 

When we read or write, we are constantly re-organizing our understanding of topics. This happened at least three times for the very blog post you're reading; the title and content has evolved dramatically over the past two weeks since I began writing it. 

I contend that **summarization is thinking**. When I am reading about a new topic, I know that I've understood it *only* when I can accurately and concisely summarize it.[^4] Robust summarization is only possible when you can grasp the big picture intuitions and connect them to the minute details. That mental organization **is** a part of the learning process. When an LLM does this organization on your behalf, then your mental muscles atrophy. 

---

[^1]: But don't worry too much about the details of this percolation theory analogy, and like all analogies, it breaks down under scrutiny. I hope you're not distracted by wondering about the order parameter or anything like that.
[^2]: I sometimes call this the *generator–discriminator asymmetry*. This is commonly used to describe how it's far easier for a GAN to discriminate between a real or generated output, than to generate some new output that can fool the discriminator. It can also be used to refer to human learners: discriminating right from wrong information is easier than correctly deducing something from scratch! (Side bar to my footnote: this also gets at why multiple choice questions are [bad for evaluating LLMs](https://www.nature.com/articles/s41598-025-26036-7)!)
[^3]: Gemini 3 is stunningly good at this. In about 60 seconds, it created a simple HTML game for my daughter to learn simple combinatorics, based on my prompt to center the game around making permutations and combinations of *N* ice cream flavors given *P* scoops of ice cream on a cone. She loved it!
[^4]: It is also incredibly easy to discern which junior researchers have truly understood a topic versus those who haven't by asking them to summarize a topic. 
---
title: 'Four ways I use LLMs'
date: 2025-04-28
permalink: /blog/2025/04/four-ways-i-use-llms/
tags:
  - llms
  - productivity
---

Large language models (LLMs) haven't upped my productivity by 10x, but they have dramatically changed the way that I work. In this post I introduce four ways that I use LLMs every day.

These techniques include *(i)* leveraging their semantic search capabilities for research exploration and surveying prior work, *(ii)* offloading mundane or tedious tasks I'd rather not spend mental energy on, *(iii)* using them as a personalized copyeditor (with careful constraints!), and *(iv)* depending on LLMs as a personal assistant to organize my work according to the [Eisenhower Matrix](https://en.wikipedia.org/wiki/Time_management#Eisenhower_method). Below, I'll walk through each of these use cases with concrete examples and some hard-won lessons about when (and when not) to rely on them.

## 1. Exploring ideas and surveying prior art

In a [recent post](https://www.lesswrong.com/s/5GT3yoYM9gRmMEKqL/p/hjMy4ZxS5ogA9cTYK), [Neel Nanda](https://neelnanda.io/) breaks down his research process into four steps: ideation, exploration, understanding, and distillation. This first step – coming up with a new idea to tackle some problem – is extremely difficult if you don't have lots of breadth in a research domain. Neel wisely suggests that junior researchers should lean on mentors/advisors during the ideation process. This is because it's difficult to choose a good problem if you don't have years of experience to survey the literature in this field.

I've been doing both astronomy and some form of machine learning for almost 15 years,[^1] so I can quickly scan over literature and evaluate whether a given paper represents a consensus view, an exciting new development, or a far-out perspective that is unlikely to gain traction. However, I do not have a very complete index of the literature. And even if I did, it would still be extremely time-consuming for me to find all of the papers that are potentially relevant to some new idea. This is where LLMs, which excel at semantic search, can provide tremendous value!

Whenever I have a new idea, I try to decompose it into smaller chunks, i.e., easier-to-understand concepts. Not only does this help me refine my mental model of the idea, but it also improves my communication – in this case, with an LLM. 

As a concrete example, I was just thinking about random matrix theory in the shower (as one does), and was wondering if we could monitor the evolution of the eigenspectra of neural network weight matrices, from initialization to training convergence. By repeating this experiment many times, we could monitor the ensemble of singular values, and perhaps learn something about the optimization dynamics or weight initialization. However, I was also pretty sure that this should have been studied before, so I mostly wanted to learn more.

I fed this into GPT Deep Research (which uses `o3`); you can see the results [here](https://chatgpt.com/share/68101fc7-1840-8001-96e9-11a54c810e16). I also ran it through Gemini 2.5 Pro, with very similar results. In 15 minutes, I went from shower thought → broken down concepts → survey of prior art. These kinds of LLM results are helpful for discovering interesting papers and fleshing out ideas, and also for deciding what kinds of random side projects I might take on. 

In this case, it turned out that this topic has been thoroughly explored, and I learned that my idea wasn't novel at all! Better to learn that now, rather than after I had invested a bunch of time.

## 2. Automating things that I know how to do

One of my favorite uses of LLMs is to bypass mundane tasks that take up valuable mental energy. For example, if I'm trying to rapidly implement some idea, then I don't want to pause and think about how to write an obfuscated regex pattern, or figure out that synonym of the word that's on the tip of my tongue, or remember the syntax for `grep`, or recall what the args of [`astropy.coordinates.SkyCoords`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html).[^2] I *could* figure these things out myself, but I've already learned them before. Many, many times. Right now I want to allocate full brainpower to the task at hand.

For tasks that are more than simple lookups or `bash` one-liners, I need to add two important corollaries:
- If I want to automate a task using a LLM, then I must know how to do it well enough to quickly and cleanly explain each of the necessary subtasks. If I cannot decompose the task into a plan of subtasks, then I must have an incomplete abstraction. **This is dangerous territory!** It means that I won't be able to evaluate the LLM outputs well enough to know if they're right or wrong. And ultimately, this indicates that I'm susceptible to wasting more time than I save.
- Alternatively, I can decide that I do not care about learning how to do this task. For example, when I [migrated my blog](https://jwuphysics.github.io/blog/2025/04/hello-world-again/) back onto Github Pages (see LLM prompt [here](https://bsky.app/profile/jwuphysics.bsky.social/post/3lnjvw2zy522w)), I decided that learning the [Jekyll](https://jekyllrb.com/docs/liquid/) and [Liquid](https://shopify.github.io/liquid/) was not worth it, and that ultimately I just wanted to get my site working.[^3] In this case, I could visually *validate* that my blog displayed correctly, and that was good enough for me.

A more general piece of advice is that LLMs are valuable for helping you stay in the zone. This is my personal take on Andrej Karpathy's (in)famous ["vibe coding" tweet](https://x.com/karpathy/status/1886192184808149383?lang=en). If you're just doing a throwaway project, or trying to fix some stupid CSS rendering issue that you honestly might just nuke and re-fork from the original template, then by all means, let the LLM run its magic. But usually, you'll want to keep a tight leash on what you let the LLM do (i.e. **you** should write down its plan, rather than let it decide for itself). Too much free rein and the LLM might overengineer some bonkers solution or start refactoring stuff. Too little, and you'll find yourself bogged down in boring stuff and losing sight of what you originally set out to do.

## 3. Copyediting my writing

I tend to write in a very non-linear fashion, which means that I frequently jump between writing my current sentence and revising the wording in previous ones. Unfortunately, that also means that I can sometimes leave a sentence completely unfinished, or use an uncommon word multiple times in a paragraph. 

Long-context LLMs are super useful for checking grammar and performing copyedits.

Note: Like many people, I have distinctive writing quirks. I tend to use a lot of parentheses coupled with the term *exempli gratia* (e.g., ...). [Unfortunately](https://www.rollingstone.com/culture/culture-features/chatgpt-hypen-em-dash-ai-writing-1235314945/), I also like to use em-dashes (—), semi-colons (;), and in-line lists like *(a)*, *(b)*, and finally, *(c)*. If left unchecked, LLMs will override my style with their own, so it's vital to maintain control over what the LLM does.

A terrible prompt:
> You are a professional writer with renowned style. Rewrite my essay draft.

A better prompt:
> Act as a copyeditor for the following essay draft. Please make a note of all spelling and grammatical mistakes. If certain words are overused, or if any sentences sound awkward or roundabout, please propose alternatives. Only make suggestions where necessary. **Do not change the overall tone or writing style.**

I request that the LLM propose edits rather than rewrite the whole thing. This forces me to manually insert edits and – while I'm pausing to do so – think about whether I actually like each suggestion.

## 4. Organizing my Eisenhower Matrix task list

This one is especially great for LLMs that can write and edit from a ledger, like Claude Artifacts or Gemini Canvas. 

I start a new conversation, give it a name like "2025-04 Tasks", and begin with a prompt like this:
> You maintain and organize my tasks in an artifact. The task list is based on the Eisenhower matrix of priority versus urgency: Q1 (urgent + important), Q2 (not urgent + important), Q3 (urgent + not important), and Q4 (not urgent + not important). You should separately track upcoming deadlines and events (date and event) in headings below. 
>
> Record today's date at the top of the document, and I'll proceed to describe tasks that have recently been completed, as well as upcoming tasks. In general I will tell you which quadrant each task belongs in. 
> 
> Detailed instructions:
> * Do not offer to do anything beyond these explicit instructions.
> * You may infer the relative prioritization within each quadrant (e.g. between Q2 tasks) based on the context that I provide. I will say something if the ordering of tasks looks incorrect. 
> * Do not break down tasks into subtasks unless I say so.
> * Tasks should be accompanied by deadlines, and if relevant, points of contact. If I describe updates to these tasks, then please modify them accordingly (don't just append details to the original task).
> * In a separate subsection, list upcoming deadlines and events in chronological order.
> * In another subsection, keep track of recently completed tasks. Two weeks after a task is completed, then remove it from the completed task list.

I've found enormous value chatting with this LLM at the beginning of each work day while I'm driving in. It helps free up my mind and allows me to jump straight into work, rather than scan emails and perform minor bureaucratic tasks (which will be safely stowed in Q3, so I don't need to stress about them). My long-horizon speculative projects reside in Q2, and I can attend to them when I'm on a [*maker's schedule*](https://www.paulgraham.com/makersschedule.html). Q1 tasks are highest priority, and this task list (which I'll leave up throughout the day) reminds me not to get distracted working on other items!

---

[^1]: Fun fact: although I majored in Physics at Carnegie Mellon University, my first internship was in ML/Robotics/ECE at the [CMU CyLab](https://www.cylab.cmu.edu/).
[^2]: Seriously, why is the argument `unit` and not `units`? 
[^3]: RIP my front-end/full-stack career.
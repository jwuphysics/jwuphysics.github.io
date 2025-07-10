---
title: 'Clear Vision, Clear Communications'
date: 2025-07-10
permalink: /blog/2025/07/clear-vision/
tags:
  - advice
  - llms
---

[The Eleven Laws of Showrunning](https://okbjgm.weebly.com/uploads/3/1/5/0/31506003/11_laws_of_showrunning_nice_version.pdf) by [Javier Grillo-Marxuach](https://en.wikipedia.org/wiki/Javier_Grillo-Marxuach) is full of useful advice for management and operations. 

A lot of this guidance is surprisingly relevant for working with large language models (LLMs). Check out this blurb:

> When you're a showrunner, it is on you to define the tone, the story, and the characters. You are NOT a curator of other people's ideas. You are their motivator, their inspiration, and the person responsible for their implementation. 
> 
> Bottom line: the creativity of your staff isn't for coming up with your core ideas for you, it's for making your core ideas bigger and better once you've come up with them. To say "I'll know it when I see it" is to abdicate the hard work of creation while hoarding the authority to declare what is or isn't good. 

This is the number one failure mode I see for people just starting to use LLMs. Inexperienced users usually give a short, poorly specified prompt, and then hope that the LLM will magically read their minds and respond with the their *intent* rather than what they've literally written in the prompt. These users are giving vague directions, and then implying "I'll know it when I see it." Sorry folks, AI isn't telepathy.

I've discussed this a bit [before](https://jwuphysics.github.io/blog/2025/04/four-ways-i-use-llms/), but here's another concrete example. 

Imagine you're excited by the new [LSST Data Preview](https://dp1.lsst.io/), and you want to try out some new research ideas. You ask ChatGPT "*What are some research questions I can answer with the new LSST data?*" It lists some [generic high-level stuff](https://chatgpt.com/s/t_686f9e5dc04c8191bc58c2a17eb995ff) that's probably lifted from a white paper summary from years ago. You think to yourself, *Wait, I don't care about all these topics, I just wanted topics in galaxy evolution. And maybe using machine learning. Oh and also I don't care about predicting photo-zs better, everybody has already been trying to do that for decades. Oh yeah and only use data that can be crossmatched with these value-added catalogs.* This is probably going to be a back-and-forth process, wasting your time, polluting the LLM context, and probably leaving you frustrated and without any good research ideas.

Let me propose a better alternative. Spend 5 minutes thinking about the essence, the specifics of what you're looking for. You might jot down your prior research ideas, examples of research topics you don't care about, extra information you know is relevant but the LLM might not index on. *Okay, I am interested in ML and galaxy evolution. I've previously worked on predicting spectroscopic information purely from photometric imaging. I also am interested in graph-based representations of galaxies and large scale structure. I don't want to just repeat what I've done before; I'm also not interested in science cases XYZ. LSST offers some competetive advantages over other wide-area surveys: transient science, u-band imaging, low surface brightness sensitivity.* Iterate on this trellis, and then let the LLM expand outward and fill gaps inward.

When working with LLMs, it is on **you** to define the tone, the core ideas, the new insights. Carefully crafting and communicating this vision is a foundational skill, useful for personal brainstorming or managing an academic research group — it certainly goes beyond just LLM prompting or showrunning!

---

Thanks to [Simon Willison's blog](https://simonwillison.net/2019/Feb/19/eleven-laws-showrunning/) — that's where I first heard about this.
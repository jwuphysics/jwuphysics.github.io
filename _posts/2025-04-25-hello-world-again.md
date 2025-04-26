---
title: 'Hello World (again)!'
date: 2025-04-25
permalink: /blog/2025/04/hello-world-again/
tags:
  - blogging
  - personal
---

Oh hey, it's me again. Yes, I'm going to try to write more blog posts. Yes, I've promised that before. Sorry. This time it'll be different.

## Why start blogging? 

After I announced the triumphant return of my blog on Twitter, [Nolan Koblischke](https://nolank.ca/) [asked](https://x.com/astro_nolan/status/1915560269096575034) what inspired me to start blogging – I'll expand on my answer here.

I started blogging during the middle of 2020 primarily for three reasons.

First, [Jeremy Howard](https://jeremy.fast.ai/) consistently advocated for blogging as a way to improve written communication skills.[^1] I had also enjoyed reading content from prolific bloggers including [Scott Alexander](https://slatestarcodex.com/), [Gwern](https://gwern.net/), [Ben Kuhn](https://www.benkuhn.net/), [Julia Evans](https://jvns.ca/), [Dan Luu](https://danluu.com/), to name a few, and I really wanted to emulate them.

Second, I wanted to establish some credibility in the subfield of astronomical machine learning. I had recently completed my PhD in Astrophysics with a dissertation entitled [*Insights on Galaxy Evolution from Studies of the Multiphase Interstellar Medium*](https://www.proquest.com/openview/0aef9b808cb10ce193098c6dd97cf06c), which was mostly about detailed multi-wavelength observations to study different galaxy populations. But by 2020, I was hoping to focus my academic career on ML for astronomy. So, it seemed like a good idea to kickstart my blog by writing [tutorials](https://jwuphysics.github.io/tags/#tutorial) on, e.g., reproducing [results from my papers](https://jwuphysics.github.io/blog/2020/05/learning-galaxy-metallicity-cnns/), [expanding on ideas](https://jwuphysics.github.io/blog/2020/08/visualizing-cnn-attributions/) that didn't amount to a paper, or [presenting topics](https://jwuphysics.github.io/blog/2020/12/galaxy-autoencoders/) that I wanted to learn more about.

Third, I needed a way to broaden my professional network. I was a newly minted PhD, six months into my first postdoc position, and then Covid struck. As an aspiring academic with almost no existing network in the ML community, I couldn't afford to miss out on this important season.[^1] So again, I turned to blogging and sharing my posts on social media. 

## Migrating to Substack

I genuinely enjoy writing new blog posts. Writing has helped me better organize my thoughts . However, I found it drudgerous to maintain my blog site. I credit [Fastpages](https://fastpages.fast.ai/) with removing the initial friction of getting my blog started, but over time, I began running into problems – technical debt, if you will. It was mostly my fault: I tried to customize the site and ended up breaking components that I didn't care to understand. Fastpages was still being updated, and apparently my early version couldn't be easily merged with the updated templates. And then Github's Dependabot started to holler at me. It was evident that my Fastpages blog would be difficult to maintain. 

Around the middle of 2021, I decided to move to Substack. This migration apparently caught the attention of [one of my readers](https://github.com/jwuphysics/blog/issues/9); when asked why I moved platforms, I responded:[^3]
> Good question [@akvsak](https://github.com/akvsak), I've thought about this too. I was using an early version of fastpages and rather than break my blog by updating, I decided to break it by migrating to Substack. I was also constantly getting flagged by dependabot for various security vulnerabilities, which I found annoying. Ultimately I'm not sure whether that was a good or bad decision to migrate!
> 
> With Substack it's easier for me to focus on writing, which I thought was more important than providing inline code (since the code exists in various repositories anyway). However, you obviously have less control than if you host your own blog on Github pages. If I had made my blog later, I probably would have used [academicpages](https://academicpages.github.io/). In any event, I think that it's more important to find a good-enough platform and just keep writing!

By late 2021, I had secured a tenure-track faculty research position at STScI, which meant that I had less time to write. (My postdoc position enabled 100% of my time for research, while my faculty-level position enabled 50% for research–comparable to a R1 professor.) Even Substack's easy-to-use interface wasn't enough to empower me to regularly continue posting. And so I only rarely posted into my Substack blog after that.

## Coming back home

In the past year, we've seen social media platforms fall apart (e.g. the [exodus of scientists from Twitter](https://www.the-independent.com/tech/twitter-academics-x-elon-musk-takeover-b2633541.html)) and restrict access (e.g., [on Reddit](https://www.theverge.com/2023/6/12/23755974/reddit-subreddits-going-dark-private-protest-api-changes)). I started posting more on [Bluesky](https://bsky.app/profile/jwuphysics.bsky.social) instead of on Twitter,[^4] and I felt some comfort knowing that I "owned" my posts on Substack. But, any of these other platforms could change their policies or close down. I could spend thousands of hours investing on one of these platforms, only for it to evaporate in a moment. As the saying goes, *don't build your house on somebody else's land*.

I wanted to make sure that I had an up-to-date blog that was agnostic to where it was hosted. Around this time, I also shifted my [home page](https://jwuphysics.github.io/) from an [HTML5up.net](https://html5up.net/) design—which I had maintained for 10 years—to a simpler [academicpages](https://github.com/academicpages/academicpages.github.io) design.

The first thing I did was to dust off my old blog posts and migrate them to my new blog. This involved updating my academicpages template page (it was about two years out of date) and converting my old blog posts (.ipynb files) and Substack posts (exported to HTML) into Markdown. [Gemini 2.5 Pro (experimental)](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-pro)[^5] was invaluable for helping me migrate my blog in less than a day. I described some of the process in [this Twitter post](https://x.com/jwuphysics/status/1915765387053977700); maybe I'll write more about it later.

Anyway, the blog is back up and running. Hope to see you back for more!

---

[^1]: He also encouraged microblogging, e.g. by posting on social media, to practice brevity and clarity in communication.  I started a [Twitter account](https://twitter.com/jwuphysics/) around the time that I launched my blog.
[^2]: My PhD advisor described the postdoc years as the peak time for research because you have enough expertise and agency to actually get work done, while still having few professional obligations (e.g. teaching, department shenanigans, or professional service).
[^3]: Amusingly, I forgot about my response here until I was writing this blog post. However, the words I said two years ago were fairly prescient: Substack offers less control than self-hosting (well, Github Pages-hosting) my blog, and academicpages seems like a more sustainable alternative than Fastpages. Both of these contributed to resurrecting my blog again!
[^4]: It seems like Twitter has lost most of its research crowd. I also find the hordes of Blue Check reply guys completely insufferable. However, it is still the best platform for AI/ML discussion. Bluesky, meanwhile, has gained quite a bit of popularity, especially amongst astronomers (see e.g. the [Astronomy Feeds](https://bsky.app/profile/astronomy.blue) curated by [Emily Hunt](https://bsky.app/profile/did:plc:jcoy7v3a2t4rcfdh6i4kza25)), but it still has less activity than AstroTwitter did in its heyday. For a while, I hung out on [Mastodon](https://mstdn.social/@jwuphysics@astrodon.social) too, but it never reached critical mass for the scientific research community.
[^5]: Gemini 2.5 Pro is by far the best LLM I've used recently. However, I want to mention one frustration: it repeatedly refused to transcribe one of my blog posts that discussed *privilege* and how that impacts success in academia. I'm not sure why it was so scared by that word, but I'm now a bit wary about censorship in Gemini models. You can read [my Bluesky thread](https://bsky.app/profile/jwuphysics.bsky.social/post/3lnkw622elk2y) for more.
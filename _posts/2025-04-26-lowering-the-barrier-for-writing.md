---
title: 'Lowering the barrier to writing more frequently'
date: 2025-04-26
permalink: /blog/2025/04/lowering-the-barrier-for-writing/
tags:
  - blogging
  - llms
  - productivity
---

These posts are fun to write. 

[Simon Willison](https://simonwillison.net/2025/Apr/25/backfill-your-blog/) also suggested backfilling the blog with previous posts made on other platforms. 

But writing new posts can be time consuming... Let's be honest: am I going to stay motivated and continue writing blog posts? **To sustain good habits, you need to make those tasks easier for your future self.**

## Making it easier to start new posts


With that in mind, I decided to write a bash script that gets my boilerplate out of the way, and enables me to immediately write a post. I decided that I wanted it to take on this usage pattern:
```sh
./start-blog-post.sh "This is the title of some future post!"
```

and then it could write the "front matter" at the top of the blog post, which should look like this:

  ---
  title: 'Lowering the barrier to writing more frequently'
  date: 2025-04-26
  permalink: /blog/2025/04/lowering-the-barrier-for-writing/
  tags:
    - blogging
    - llms
    - productivity
  ---

Now, I could put together some simple script to write most of boilerplate for me. But there's some soft logic in here: for example, I might want to abbreviate the title or remove special characters for the permalink. And I'll also want to think a bit creatively to come up with good tags. So what's the best way to code this up?

## Tasking LLMs to write boilerplate text

Soft logic sounds like a job for a local LLM! I will be using the Gemma 3 27B model with 4-bit Quantization-Aware Training (download on Ollama [here](https://ollama.com/library/gemma3:27b-it-qat)), which can fit into ~20 GB of memory and run in under 10 seconds on a GPU.[^1] 

So I decided to use the simple [`llm`](https://llm.datasette.io/en/stable/) CLI tool come up with a short title, invent a few candidate tags, and then copy over the rest of the boilerplate.

Again, this is just meant to get things started. It's okay if the LLM makes mistakes![^2] I'll still give the front matter a glance, and probably change a few of the tags. While writing this post, I decided to change the full title (that I originally specified)!

The full code I used to initiate this post can be found in this [Github Gist](https://gist.github.com/jwuphysics/99cd3fe53719933328038e748721eeba).

## Making it easier 

[^1]: I strongly suspect that lightweight instruction-following models like the 3.8B parameter [phi4-mini](https://ollama.com/library/phi4-mini) or the 8B parameter [Llama 3.1 model](https://ollama.com/library/llama3.1) should do a great job, too.
[^2]: During early testing, one of the models (Mistral 3.1 24B, I think), insisted on wrapping the output with triple-back quotes and explicitly denoting it as a `yaml` code block.
---
title: 'Lowering the barrier to writing more frequently'
date: 2025-04-26
permalink: /blog/2025/04/lowering-the-barrier-for-writing/
tags:
  - blogging
  - llms
  - productivity
---

Writing these posts is fun but time-consuming. How can I stay motivated enough to post consistently? **To sustain a habit of writing, I'll need to create an easier path for my future self.**

## Introduction

I've always got dozens of random ideas marinating in my head, and it's only after I start talking or writing about these ideas that they really become clear. Half of them are truly awful, by the way. Some of them I'm really proud of. But it's hard to know what's good and what's slop until I take the time to spell out and reorganize my thoughts.

Writing is perfect for this. However, my previous blogging habit lapsed partly because getting started was a hurdle. Even with time available, switching gears to set up a new post (e.g. getting the boilerplate headers right) often drained my initial enthusiasm before I even began writing. 

Switching to Substack helped with this, for a while. But as I mentioned in my [last post](https://jwuphysics.github.io/blog/2025/04/hello-world-again/), my blog has been fully returned to my website. So again, I've got to get over the mental barrier of formatting the post before I lose interest in writing.

## Making it easier to start new posts

With that in mind, I wrote a bash script that lets me start writing immediately. I wanted the script to follow this usage pattern:
```sh
./start-blog-post.sh "Lowering the barrier to writing more frequently"
```

and then it would create a new markdown file with a Jekyll style blog post, and pre-fill the "front matter" at the top of the blog post, which should look like this:

```yaml
---
title: 'Lowering the barrier to writing more frequently'
date: 2025-04-26
permalink: /blog/2025/04/lowering-the-barrier-for-writing/
tags:
  - blogging
  - llms
  - productivity
---
```

Now, I could write a simple script to handle most of the boilerplate. But there's some soft logic in here: for example, I might want to abbreviate the title or remove special characters for the permalink. And I’ll also want help generating relevant tags. So what's the best way to code this up?

## Tasking LLMs to write boilerplate text

Soft logic sounds like a job for a large language model (LLM)! I will be using the Gemma 3 27B model with 4-bit Quantization-Aware Training (download on Ollama [here](https://ollama.com/library/gemma3:27b-it-qat)), which can fit into ~20 GB of memory and run in under 10 seconds on a GPU.[^1] 

"So I used the simple [`llm`](https://llm.datasette.io/en/stable/) CLI tool to suggest a short title, generate relevant tags, and populate the rest of the boilerplate.

Again, this is just meant to get things started. It's okay if the LLM makes mistakes![^2] I'll still give the front matter a glance, and probably change a few of the tags. In fact, while writing this post, I decided to change the full title that *I* originally specified!

The full code I used to initiate this post can be found in this [Github Gist](https://gist.github.com/jwuphysics/99cd3fe53719933328038e748721eeba).

---

[^1]: I strongly suspect that lightweight instruction-following models like the 3.8B parameter [phi4-mini](https://ollama.com/library/phi4-mini) or the 8B parameter [Llama 3.1 model](https://ollama.com/library/llama3.1) should do a great job, too.
[^2]: During early testing, one of the models (Mistral 3.1 24B, I think), insisted on wrapping the output with triple-back quotes and explicitly denoting it as a `yaml` code block. I also considered using the LLM to directly output just the short-title and the tags according to some JSON schema, but then I'd need to invoke some additional tools like `jq` to parse the output, and it seemed like a lot of extra work. In any event, the local LLM has performed good enough for me.
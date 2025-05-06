---
title: 'Accelerate the Boring Stuff with AI'
date: 2025-05-06
permalink: /blog/2025/05/accelerate-boring-stuff/
tags:
  - llms
  - productivity
---

Need to perform a boring, repetitive task? Even if it can't be fully automated, you may be able to dramatically speed up your task by *partially* automating it! Simply use a LLM to vibe code a throwaway app to help accelerate your mindless task. (I'm serious.)

---

Yesterday, I needed to build a [golden sample](https://jwuphysics.github.io/blog/2025/04/constructing-golden-sample/) for evaluations — undoubtedly one of the highest-leverage tasks you can do when working in real-world AI/ML.

But compiling this sample was dreadfully slow and monotonous. I glanced at the time. *Gotta pick up my kids in an hour.* At my current pace of annotating data, I won't be able to finish by the end of the day.

More than half of my time on this task was spent clicking around, opening and closing files, and manually entering data into a spreadsheet. Because it was so mindnumbingly boring, I found myself too easily distracted from annotating the data. 

*There must be a better way!*[^1]

Indeed, I recognized some potential for automation: I had been systematically going through a list of LLM outputs, finding the original source documents, and annotating binary labels based on my own judgment. It was very easy to describe my task, and I could even specify the usage patterns, data inputs, expectations for internal data structure representations, outputs, and potential gotchas. I still needed to be able to scroll around and inspect the PDF, and enter a `y` or `n` into an overlay window that records these labels. *Maybe I could simply ask a LLM to write a terrible Python app for me?*

Armed with newfound resolve, I turned these requirements into a prompt and chucked it into Gemini 2.5 Pro.

Within five minutes, I had created and iterated on a basic script to help me annotate data. I paid zero attention to formatting, comments, code quality, etc. All I did was verify that the code could save a single row of labels and log outputs, and, *Bob's your uncle*[^2] — I was in business. In less than half an hour, I had finished labeling my data set!

---

For posterity, I include my prompt:

```
Help me write a python app for annotating data. The app should be able to:
- receive as input a list of arxiv ids like (YYMM.NNNNN, e.g. 2501.09854) and turns each into a URL to the v1 PDF, like this: https://arxiv.org/pdf/2501.09854v1
- iterate through the list of PDF URLs
- allow a user to inspect each PDF in a new browser window
- enable a user to enter "y" or "n" in a box that floats on top of all other windows.
- as soon as either a "y" or "n" input is received, that input is recorded; then the PDF window will close and the next URL will open
- when all elements have been cycled through, then save a CSV with two columns, the arxiv_id in the first column, and a boolean (True for "y" and False for "n") in the second column. (Save this with some name but if the file exists, then choose a distinguishing name so we don't overwrite the previous one)
```

and Gemini's response can be found [here](https://gist.github.com/jwuphysics/39d525ecbf357e9ca03b7cfd98695988).

---
[^1]: I always hear this in [Raymond Hettinger's voice](https://www.youtube.com/watch?v=wf-BqAjZb8M). If I didn't have [aphantasia](https://en.wikipedia.org/wiki/Aphantasia), I'd probably also imagine him banging a table, too.
[^2]: I'm always on the lookout to drop this amazing idiom into everyday conversation.
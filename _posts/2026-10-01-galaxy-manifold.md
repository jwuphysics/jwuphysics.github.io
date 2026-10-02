---
title: 'The Galaxy Manifold'
date: 2026-10-01
permalink: /blog/2026-10-01/galaxy-manifold/
tags:
  - galaxies
  - llms
  - visualization
---

Last Thursday I was surprised that my calendar showed—for the first time in several weeks—zero meetings! I told everyone that I was taking the day off and decided to go hiking. By the time I crossed the Mile 2 marker of the [NCR Trail](https://en.wikipedia.org/wiki/Torrey_C._Brown_Rail_Trail), I was subconsciously thinking about [students and education](https://jwuphysics.github.io/blog/2025/12/learning-with-llms/) and [AI in science/math](https://openai.com/index/navier-stokes-solution/) and the [accelerating pace of research](https://www.anthropic.com/institute/measuring-pace-of-ai-development). And I had to ask myself, *what was the most vital part of my education as a PhD student?*  

The answer was obvious: **Looking at data.**[^1] Visualizing data to build all of the astrophysical intuitions I would need to do my profession well. Wrangling data to understand correlations, selection biases, spurious signals, astrophysics. Testing data by fitting physical and statistical and machine learning models. 

It became obvious to me that AI tools could **enrich astronomical data** by joining together datasets, presenting them in new ways, and empowering users to navigate the datasets more fluidly. Most importantly, AI could build out such tools *without* compromising the learning process for students. 

I had also come up with the perfect use case: the multitude of galaxy scaling relations. Researchers typically characterize these scaling laws that explain how galaxy properties are correlated as they grow, such as the [mass-metallicity relation (MZR)](https://ui.adsabs.harvard.edu/abs/2004ApJ...613..898T/abstract), or the [size-mass relation](https://ui.adsabs.harvard.edu/abs/2003MNRAS.343..978S/abstract), or the [star forming "main sequence" of galaxies](https://ui.adsabs.harvard.edu/abs/2014ApJS..214...15S/abstract). But these can also be combined together, e.g., forming the [fundamental plane](https://ui.adsabs.harvard.edu/abs/1987ApJ...313...59D/abstract) or the [fundamental metallicity relation](https://ui.adsabs.harvard.edu/abs/2010A%26A...521L..53L/abstract)! Intuitively, we know this means that galaxies actually live on a low-dimensional manifold even when we plot them in a high-dimensional space.

## Designing the visualization tool

After a few hours of hiking and mulling over these thoughts, I couldn't wait to build this tool. (And if you can't wait to see it, just skip down to the next section, or [view it directly](https://galaxy-manifold.github.io/)!) I had settled on some core design principles, primarily aimed at *enhancing astronomers' intuition* rather than *replacing it with AI*:
* **Center the data.** Focus on the galaxy manifold, letting users see and learn intuition based on what they see. Do not over-explain or offer interpretations. 
  * It should be easy to show three parameters on the *x*, *y*, and color axes.
  * We should be able to seamlessly display galaxy image cutouts. Those cutouts display a ton of visual information that cannot be easily summarized by a small number of parameters.
* **Empower navigation.** Make it easy for the user to navigate the data manifold, whether that's through ergonomic scrolling, panning, or rotation through this high-dimensional galaxy manifold.
  * One example here is *The Grand Tour* ([credit to Asimov!](https://epubs.siam.org/doi/10.1137/0906011)), a mode that begins randomly rotating through *2D* views in a high-dimensional space of galaxy properties.
  * You can use a lasso tool to draw a selection of galaxies and show all of their image cutouts and/or the subsample's statistics over all parameters.
* **Equip power users.** Unlike many apps that solely focus on simplified user interfaces but present only a shallow view, I wanted the Galaxy Manifold web app to be even more powerful if you knew more about galaxy evolution. 
  * For example, you can impose multiple filters on the data while cycling through multiple views. 
  * There is a feature to minimize the scatter along the *y* axis by taking a linear combination of features (projected along the *x* axis). This is analogous to how the fundamental metallicity relation was discovered.
  * There are keyboard shortcuts!
* **Connect data with astrophysics.** Data provenance, empirical relations, and theoretical predictions should all be shown side by side. 
  * The Sloan Digital Sky Survey (SDSS) [Main Galaxy Sample](https://ui.adsabs.harvard.edu/abs/2002AJ....124.1810S/abstract) is perfect for a few reasons: it is well-studied, it has a homogeneous selection criterion (*r* < 17.78), it has both imaging and optical spectra, and it is mostly limited to low-redshift objects (which imposes some bound on the rest-frame wavelengths and angular sizes). 
  * An important feature is being able to crossmatch this large SDSS sample against small, important galaxy surveys and data products (e.g., radio-wavelength catalogs, environmental catalogs, citizen science morphology labels, etc).
  * I thought it was vital to show image cutouts, with an "inspector" service that links to other surveys (like DESI Legacy Imaging Surveys and SDSS cutouts and spectra), but in a way that was snappy and didn't require loading GB of data. I instead chose random representative images using K-means and compressed them so that they can load quickly.

With all of these ideas in mind, and many more,[^2] I turned to the freshly released Claude Opus 5.5 and delivered my instructions. 

## And lo, the Galaxy Manifold

![A view of the default Galaxy Manifold webpage.]({{ site.baseurl }}/images/apps/galaxy-manifold.jpg)

There is no single canonical way to view the [Galaxy Manifold app](https://galaxy-manifold.github.io/), so all I can say is that you gotta play around with it. Not sure where to start? Fine, I'll hold your hand; here are some instructions (geared for computer use, not the mobile view).
1. Open up the [default view](https://galaxy-manifold.github.io/#p=sfms&c=D4000&v=-5.245,2.4477,-2.1458,2.1786), which should be on the star-forming main sequence, colored by the 4000 Angstrom break.
2. Press `m`, or look at the toggle in the lowest, left-most corner. This will show image cutouts. 
3. In that bottom panel, look to the right, past the colorbar, until you see a dial and another toggle. Click on the dial and slide up/down to adjust the size of the cutouts.[^3] Click on the toggle to go from apparent sizes to physical galaxy sizes.
4. Let's revert to the grid-points view because this gives us our color axis back. (Remember you can just press `m` again.) Let's also toggle off the literature scaling relations by pressing `b`. 
5. Try selecting a subpopulation of galaxies using the lasso tool, which you can do in the lower right-side corner or by pressing `l`. This opens up the selector panel on the right-hand side: you can see image cutouts, as well as galaxy property distributions along all the different axes! Press `esc` to remove the selection, or find the null symbol and click it. Press `l` again to turn off the lasso tool.
6. Did I mention that you can just hover your mouse over a point to see image cutouts pop up? Clicking one will bring up the inspector tool, which shows a live image cutout from the [Legacy Imaging Surveys](https://www.legacysurvey.org/viewer) viewer. If you scan along the upper toolbar of the inspector view, then you will also find links to the [SDSS SkyServer spectrum viewer tool](https://skyserver.sdss.org/dr20/SearchTools/SQS).
7. By now you must be wondering what that circular-looking compass tool is in the lower left corner. If you guessed that it sets the axes of the plot (i.e., [by default](https://galaxy-manifold.github.io/#p=sfms&c=D4000&v=-5.1594,2.4162,-2.1458,2.1786) plotting SFR vs stellar mass), then you'd be correct! You can actually grab the handle (the colored circle) of one of those axes and drag it around to start rotating it. This is how you navigate and rotate in the displayed dimensions.
8. See those other small colored circles displayed around it? Pick one—let's say the Dn4000 break strength—and drag it onto the compass system. Now you have three dimensions displayed at once ([example](https://galaxy-manifold.github.io/#f=NyJScQAAT88AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD8ouU4AAALQwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA&pp=sfms&c=D4000&o=ta&v=-3.7568,3.8188,-1.5393,2.7851&sp=0.323))! A handy way to explore here is to drag one of these three axes using small movements, causing a sort of parallax to help guide the eye.
9. Want to quickly change the *x*, *y*, or color axes? Click on one of the galaxy properties from the left-side, and select the axis you want.
10. If you color by an axis that has lots of missing values, then they'll show up as a beige color. No worries! You can press `v` or click on the filter symbol by the color bar to remove them.
11. Going back to the compass, you should try clicking the "play" button. This starts the grand tour: rotating randomly through the high-dimensional galaxy manifold! You can add properties to the grand tour through the left sidebar. (To start/pause the tour, you can also press `space`, but this doesn't work reliably on all browsers.)
12. Want to see the known scaling relations? Click one of the preset views defined in the top row.
13. Feel like changing the redshift range? The middle of the bottom panel will help you out. You can drag the left and right edges to define a redshift interval, and then click and drag that distribution to shift it. Double click it to reset it. (In fact, double clicking is a general reset strategy: double-clicking the dials will bring them to their defaults, and double-clicking the handle on the compass will remove that axis from the manifold projection.)

Didn't quite follow along? You can press `?` or click the information button in the top right corner for lots more details! There are lots more features that I didn't describe here; hopefully you'll enjoy all the little easter eggs.

## Go forth and do science

A surprising number of folks have already told me that they are fans of the Galaxy Manifold, and a not insignificant fraction have remarked that *this must've taken you so long to build*. On one hand, the app is based on ideas and intuitions that have been marinating in my brain for over a decade.[^4] On the other hand, I took a day off to go hiking, realized that I had to implement this idea, sent it off to Claude at 2pm, then went to [Max's Taphouse](https://maxs.com/) to taste test this year's Oktoberfest beers,[^5] and finally iterated on it during some spare time that evening in between cooking dinner and putting my kids to bed. All that to say, the entire process took less than a day. I had a working prototype within four hours of firing off my first prompt to Claude, and I "shipped" the web app in about eight hours.

My dream is that this tool will help astronomers spend more time understanding their data. I don't care if it makes you more "productive" or helps you write more papers. I care if it allows you to more deeply understand the manifold and physical scaling relations that govern how galaxies grow.

Tell me if you discover anything cool!

---
[^1]: As an observational astronomer, my data were telescope observations. But I think that "looking at data" is essential for theorists as well—it's just that their *data* are more like ideas or symbolic expressions rather than detector pixels or galaxy spectra.
[^2]: You can see these in the Github repository's [DESIGN.md](https://github.com/galaxy-manifold/galaxy-manifold.github.io/blob/main/DESIGN.md) page, which was spelled out based on an "interview" conducted by Claude Opus 5.5.
[^3]: This is the one major design flaw that needs to be fixed. The dial looks nice with the current theme, but the navigation is actually an up/down slider. Sigh.
[^4]: I remember telling one of my collaborators many years ago that my post-tenure ambition was to write down the true intrinsic dimensionality of the galaxy manifold. [This isn't a ground-breaking idea](https://ui.adsabs.harvard.edu/abs/2008Natur.455.1082D/abstract), but I think we're one step closer now to realizing it!
[^5]: Unsurprisingly, Ayinger is very hard to beat.
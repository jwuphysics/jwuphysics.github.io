---
permalink: /apps/
title: "Apps"
excerpt: "John's web apps"
author_profile: true
---

<style>
  .apps-intro { margin-bottom: 1.5em; }
  .apps-featured {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.5em;
    margin: 1em 0 2.5em;
  }
  .app-card {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--global-border-color);
    border-radius: 8px;
    overflow: hidden;
    background: var(--global-bg-color);
  }
  .app-card > a.app-shot {
    display: block;
    aspect-ratio: 16 / 10;
    overflow: hidden;
    border-bottom: 1px solid var(--global-border-color);
  }
  .app-card > a.app-shot img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: top left;
    transition: transform 0.3s ease;
  }
  .app-card > a.app-shot:hover img { transform: scale(1.03); }
  .app-card .app-body { padding: 0.9em 1.1em 1.1em; }
  .app-card h3 { margin: 0 0 0.15em; font-size: 1.15em; }
  .app-card h3 a { text-decoration: none; }
  .app-url {
    display: block;
    font-size: 0.75em;
    color: var(--global-text-color-light);
    margin-bottom: 0.6em;
  }
  .app-card p { font-size: 0.9em; margin: 0; }
  .apps-more { list-style: none; margin: 0.5em 0 0; padding: 0; }
  .apps-more li {
    padding: 0.6em 0;
    border-top: 1px solid var(--global-border-color);
    margin: 0;
  }
  .apps-more li:first-child { border-top: none; }
  .apps-more li:last-child { border-bottom: 1px solid var(--global-border-color); }
  .apps-more .app-name { font-weight: bold; }
  .apps-more .app-blurb { font-size: 0.9em; }
  .app-blurb { color: var(--global-text-color-light); font-style: italic; }
</style>

<p class="apps-intro">
  Here are some web apps I've built with the help of AI tools, mostly Claude.
</p>


<div class="apps-featured" markdown="0">

  <div class="app-card">
    <a class="app-shot" href="https://galaxy-manifold.github.io/">
      <img src="/images/apps/galaxy-manifold.jpg" alt="Galaxy Manifold" loading="lazy">
    </a>
    <div class="app-body">
      <h3><a href="https://galaxy-manifold.github.io/">Galaxy Manifold</a></h3>
      <span class="app-url">galaxy-manifold.github.io</span>
      <p class="app-blurb">An astrophysics visualization tool that I wish I had back when I was a grad student.</p>
    </div>
  </div>

  <div class="app-card">
    <a class="app-shot" href="https://ike-matrix.github.io/">
      <img src="/images/apps/ike-matrix.jpg" alt="Ike: an Eisenhower matrix with tasks sorted into four quadrants by urgency and importance. (You can thank Opus 5.5 for these made-up examples.)" loading="lazy">
    </a>
    <div class="app-body">
      <h3><a href="https://ike-matrix.github.io/">Ike</a></h3>
      <span class="app-url">ike-matrix.github.io</span>
      <p class="app-blurb">I couldn't find a good Eisenhower Matrix app so I made my own.</p>
    </div>
  </div>

</div>


<ul class="apps-more" markdown="0">
  <li>
    <a class="app-name" href="https://arithmos-game.github.io/">Arithmos</a>
    <span class="app-blurb">Educational arithmetic game for ~6 year old kids. Can log in to record high scores.</span>
  </li>
  <li>
    <a class="app-name" href="/apps/geometric-grids/">Geometric Grids</a>
    <span class="app-blurb">A super simple grid coloring app.</span>
  </li>
  <li>
    <a class="app-name" href="/apps/hundred-grid/">Hundred Board</a>
    <span class="app-blurb">Visualize divisors for integers up to 100.</span>
  </li>
  <li>
    <a class="app-name" href="/apps/md-deck/">md deck</a>
    <span class="app-blurb">Turns markdown into a slide deck.</span>
  </li>
  <li>
    <a class="app-name" href="/apps/tasto/">Tasto</a>
    <span class="app-blurb">Cello fingerboard app.</span>
  </li>
  <li>
    <a class="app-name" href="/apps/terra/">Terra Defense</a>
    <span class="app-blurb">A tower defense/asteroids-like game.</span>
  </li>
</ul>

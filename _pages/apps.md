---
permalink: /apps/
title: "Apps"
excerpt: "Apps"
author_profile: true
---

{% assign files = site.static_files | sort: 'path' %}
{%- for file in files %}
{%- if file.name == 'index.html' %}
{%- assign slug = file.path | remove_first: '/apps/' | remove: '/index.html' %}
{%- if file.path contains '/apps/' %}
{%- unless slug contains '/' %}
- [{{ slug }}](/apps/{{ slug }}/)
{%- endunless %}
{%- endif %}
{%- endif %}
{%- endfor %}

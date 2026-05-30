/* Add visible permalink anchors to headings in post/page content.
 *
 * Context: PR #1 added this logic to assets/js/_main.js, but the site only loads
 * assets/js/main.min.js via _includes/scripts.html, so the code never executed.
 * This file is a normal (published) asset and is explicitly included.
 */

(function addHeadingPermalinks() {
  const headings = document.querySelectorAll(
    ".page__content h2, .page__content h3, .page__content h4"
  );

  headings.forEach((h) => {
    // Skip if already has an explicit anchor
    if (h.querySelector(".header-anchor")) return;

    // Ensure there's an id to link to (kramdown usually provides this)
    if (!h.id) {
      const text = (h.textContent || "").trim().toLowerCase();
      const slug = text
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "");
      if (slug) h.id = slug;
    }
    if (!h.id) return;

    const a = document.createElement("a");
    a.className = "header-anchor";
    a.href = `#${h.id}`;
    a.setAttribute("aria-label", "Permalink");
    a.textContent = "#";
    h.appendChild(a);
  });
})();

# SQL Agent — Class Presentation

Two formats included:

- **`index.html`** ⭐ recommended — Reveal.js custom build with Apple/Claude aesthetic, Mermaid diagrams, syntax highlighting, clickable links, image support
- **`slides.md`** — Marp Markdown source (alternative; can export to PPTX/PDF)

Both share the same content (19 slides, ~12 minutes, English).

---

## Open the HTML deck

```bash
open presentation/index.html
```

That's it — no build step needed. It loads Reveal.js, Mermaid, and the syntax highlighter from CDNs.

### Navigation

| Key | Action |
|---|---|
| `→` / `space` | next slide |
| `←` | previous slide |
| `f` | fullscreen |
| `s` | speaker notes view |
| `o` | overview (zoom out) |
| `esc` | exit fullscreen / overview |

### URL fragments

The deck uses URL hashes — you can link directly to a slide:
- `index.html#/0` → title
- `index.html#/14` → app screenshot
- `index.html#/17` → demo

---

## Adding screenshots / images

The slide on **"The app"** (#15) has a placeholder. Drop screenshots into `presentation/images/` and update the relevant `<div class="media-placeholder">` to:

```html
<div class="media">
  <img src="images/app-overview.png" alt="SQL Agent UI">
</div>
```

Suggested screenshots to add:

| File name | What it shows |
|---|---|
| `images/app-overview.png` | The full app: dataset on left, chart + narration on right |
| `images/app-dark.png` | Dark mode of the same view |
| `images/hf-space.png` | The HF Space landing page |
| `images/hf-models.png` | The three model cards on Hugging Face |
| `images/loss-curve.png` | (optional) replace the inline xychart with a real screenshot of training loss |

---

## Export to PDF or PowerPoint

### From the HTML deck (Reveal.js)

Reveal.js has built-in PDF export via the print stylesheet:

1. Open `index.html` in Chrome
2. Append `?print-pdf` to the URL: `index.html?print-pdf`
3. Print → Save as PDF
4. Set paper size to **Landscape A4** or **Letter**, margins to **None**

### From the Marp deck

```bash
npx @marp-team/marp-cli@latest slides.md --pdf --allow-local-files
npx @marp-team/marp-cli@latest slides.md --pptx --allow-local-files
```

---

## Presentation tips for the team

- **Slide 18 (live demo)** is the climax — rehearse the demo end-to-end at least 5×
- **Backup**: record a 60-second screen capture of the demo in case Wi-Fi or quota fails
- **Slide 1**: pause 3 seconds after revealing — let the title land
- **Engineering notes (slide 16)**: tell each one as a quick war story, not a bullet point
- **Closing (slide 19)**: deliver the line with confidence — *"Three fine-tunes. Thirty dollars. Thirteen and a half hours."*
- **Speaking handoffs**: roughly 6 min per teammate if 2 people, 4 min if 3

## What to bring

1. Laptop with `presentation/index.html` open offline (CDNs are loaded — works without Wi-Fi after first load)
2. Logged-in HF browser tab pointing at the Space (test quota beforehand!)
3. Pre-recorded screen capture of the demo as backup
4. A small CSV ready (Coffee or Retail dataset, ~250 rows for fast demo)
5. 3 questions you've rehearsed and know produce good charts

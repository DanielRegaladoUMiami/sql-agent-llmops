# SQL Agent — Class Presentation

Slide deck written in **Marp Markdown** with an Apple × Deloitte aesthetic
(off-white background, near-black ink, Deloitte signature green accent
`#86BC25`, SF Pro typography, generous whitespace, Mermaid diagrams).

- 17 slides
- ~12 minutes
- Storytelling-first: Title → Agenda → Why → Data → Hub → Training → Architecture → Demo

## How to view / export

### Option 1 — VS Code (easiest)

1. Install the **"Marp for VS Code"** extension
2. Open `slides.md`
3. Click the Marp icon top-right of the editor → preview
4. Export → `Markdown: Export slide deck...` → choose PDF or PPTX

### Option 2 — Marp CLI

```bash
npm install -g @marp-team/marp-cli

# To PDF
marp slides.md --pdf --allow-local-files

# To PPTX (editable in PowerPoint / Keynote)
marp slides.md --pptx --allow-local-files

# To HTML (self-contained presentation page)
marp slides.md --html --allow-local-files
```

### Option 3 — Marp Web

Paste the contents of `slides.md` into https://web.marp.app/ and export.

## Tips for the live presentation

- **Slide 14 (live demo)** is the climax — rehearse 5× that the demo works
- **Backup**: record a 60-second screen capture of the demo flow in case Wi-Fi or quota fails on the day
- **Slide 1**: pause 3 seconds after appearing — let the question land
- Speak in first person: *"I trained"*, *"I curated"*, *"I shipped"*
- Engineering notes (slide 13) work best as **war stories**, not bullet points
- Closing line (slide 17): say it with confidence — *"Three fine-tunes, thirty dollars, thirteen and a half hours."*

## What to bring

1. Laptop with the Space open + signed in to Hugging Face (test quota beforehand)
2. Pre-recorded demo video as backup
3. A small CSV ready (Coffee or Retail dataset, ~250 rows for fast demo)
4. 3 questions you've rehearsed and know produce good charts

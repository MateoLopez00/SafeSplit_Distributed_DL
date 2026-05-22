# Building the PDF

This folder contains the official ICML 2025 style files (`icml2025.sty`, `icml2025.bst`, supporting packages), `main.tex`, `references.bib`, and figure assets under `figures/`.

## Local build (requires a LaTeX distribution)

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `hyperref` conflicts with the style file, switch the preamble to `\usepackage[nohyperref]{icml2025}` and load `url` separately, as described in `icml2025.sty`.

## Overleaf

1. Zip the entire `project_report` folder (keep paths flat inside the archive).
2. Upload to Overleaf as a new project.
3. Set the main document to `main.tex`.
4. Compile with pdfLaTeX and BibTeX.

Replace the placeholder author block in `main.tex` with your group names and affiliations before final submission.

## Recent fixes (project PDF)

- Title uses `\texorpdfstring{...}{...}` so PDF bookmarks avoid `\\` and hyperref stops complaining about PDF strings.
- `\phantomsection` precedes `\printAffiliationsAndNotice{}` to reduce empty-anchor warnings with hyperref.
- Wide tables use `adjustbox`; the overview figure uses `\resizebox{\linewidth}{!}{...}` on the TikZ picture.

After `pdflatex`/`bibtex`, confirm **total PDF pages** match your course rule (often seven to eight pages **including** references unless stated otherwise).

---
name: generate-tikz-pdf
description: Compiles TikZ diagrams in .tex files to PDF using pdflatex. Use when the user wants to render TikZ figures, compile LaTeX diagrams, or convert .tex files with TikZ code to PDF.
---

# Render TikZ Diagrams to PDF

Compile `.tex` files containing TikZ diagrams to PDF using `pdflatex`.

## Prerequisites

```bash
sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-pictures
```

## Compile Command

```bash
pdflatex -interaction=nonstopmode -output-directory=<output_dir> <file.tex>
```

Run **twice** if the document uses cross-references or labels.

## Crop to Diagram Bounds

Use the `standalone` document class in your `.tex` file to auto-crop to content bounds:

```latex
\documentclass[tikz,border=10pt]{standalone}
```

## Convert PDF to PNG (optional, for verification)

```bash
# Requires poppler-utils
sudo apt-get install -y poppler-utils
pdftoppm -png -r 300 diagram.pdf diagram
# Produces diagram-1.png
```

## Flag Reference

| Flag | Purpose |
|------|---------|
| `-interaction=nonstopmode` | Don't stop on errors |
| `-output-directory=<dir>` | Output directory for PDF |
| `-halt-on-error` | Stop at first error |
| `-jobname=<name>` | Override output filename |

## Best Practices

- Use `standalone` class for individual figures (auto-crops to content)
- Use `\usetikzlibrary{...}` for arrows, positioning, shapes, calc, fit, backgrounds
- Place `.tex` source files in `Static/tikz/`
- Output PDFs to `artifacts/output/`
- **Never read or inspect high-DPI PNGs** (e.g. 200+ DPI). Always decrease the DPI first (e.g. to 72), re-export, and read the low-DPI version for verification. Reading large high-DPI images may exceed context limits and crash the session.

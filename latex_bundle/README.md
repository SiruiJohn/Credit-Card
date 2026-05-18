# LaTeX Bundle Usage

This folder is a self-contained bundle for editing in an external LaTeX editor.

## Structure
- `main.tex`: main report file
- `figures/`: all referenced figures used by `main.tex`
- `refs/references.bib`: BibTeX file placeholder for references

## Compile
Run in this folder:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

## Notes
- Image paths in `main.tex` are relative to this folder (`figures/...`), so moving this folder elsewhere keeps links valid.
- If you later add bibliography citations, keep your `.bib` entries in `refs/references.bib`.

# Paper: sources and rebuild instructions

This directory holds the paper sources. LaTeX build outputs
(`.aux`/`.bbl`/`.blg`/`.dvi`/`.out`/`.ps`/`.synctex.gz`, `*-eps-converted-to.pdf`,
editor `*~` backups) are untracked and ignored — only sources plus the
distributable `prisoners.pdf` are committed.

## Sources

- `prisoners.tex` — main document (LLNCS class, `llncs.cls`),
  bibliography style `ieeetr.bst`, references in `prisoners.bib`.
  Inputs: `embedPairwiseStatic.tex`, `embedNeighbourhoodStatic.tex`,
  `embedFigStaticAvg.tex`, `embedFigQLearningAvg.tex`
  (`embedStaticFig1/2.tex` are unused variants kept for reference).
- `pairwise.tex` / `neighbourhood.tex` — gnuplot `epslatex` figure bodies
  overlaid on `pairwise.eps` / `neighbourhood.eps`.
- `*.plt` — gnuplot scripts; `*.txt` — the plotted data
  (`avgStatic100.txt`, `neighbourhoodData.txt`, `pairwiseData.txt`, ...).
- `prisoners.pdf` — last built distributable (tracked so the paper reads
  without a LaTeX install).
- `Cooperation in N-Person ... .pdf` — reference literature (a source,
  not a build output).
- `deep_research_results.txt` — research notes feeding the paper.

## Rebuilding

You need `gnuplot` plus a LaTeX distribution (`pdflatex`, `bibtex`).
`avg.tex`/`avgQ.tex` are **generated** — they are not committed, so the
gnuplot step must run first:

```bash
cd paper
gnuplot avgStatic.plt      # writes avg.tex (+ avg.eps)
gnuplot avgQLearning.plt   # writes avgQ.tex (+ avgQ.eps)
pdflatex --shell-escape prisoners   # --shell-escape lets epstopdf convert the .eps figures
bibtex prisoners
pdflatex --shell-escape prisoners
pdflatex --shell-escape prisoners
```

(`neighbourhood.tex`/`pairwise.tex` are already committed, so their `.plt`
scripts only need re-running if the underlying `.txt` data changes.)

## After rebuilding

- `prisoners.pdf` is the only build product worth committing (it is the
  tracked distributable); everything else the build writes is ignored.
- If the PDF looks stale, rebuild from the steps above rather than editing
  it — it is generated.

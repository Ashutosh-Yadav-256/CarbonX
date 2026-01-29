# CarbonX Research Paper

## Files

- `carbonx_paper.tex` - Main LaTeX paper (IEEE format)

## Compiling the Paper

### Option 1: Overleaf (Recommended)
1. Go to [overleaf.com](https://www.overleaf.com)
2. Create new project → Upload Project
3. Upload `carbonx_paper.tex`
4. Click "Recompile" to generate PDF

### Option 2: Local LaTeX
```bash
# Install TeX Live or MiKTeX first
pdflatex carbonx_paper.tex
pdflatex carbonx_paper.tex  # Run twice for references
```

### Option 3: VS Code
1. Install LaTeX Workshop extension
2. Open `carbonx_paper.tex`
3. Press Ctrl+Alt+B to build

## Paper Structure

| Section | Description |
|---------|-------------|
| Abstract | 150 words summarizing key results |
| Introduction | Problem, motivation, contributions |
| Related Work | Green AI, efficient inference, carbon-aware computing |
| Methodology | Problem formulation, algorithms, architecture |
| Experiments | Setup, datasets, baselines, results |
| Discussion | Trade-offs, limitations, future work |
| Conclusion | Summary of contributions |

## Key Results (Pre-filled)

- **55% carbon reduction** vs always-large baseline
- **73.2% accuracy** on ML benchmarks
- Model distribution: Small 39%, Medium 39%, Large 22%

## Customization

1. **Author info**: Update `\author{}` section
2. **Add figures**: Copy from `experiments/figures/` and use `\includegraphics`
3. **Expand datasets**: Add more samples to improve experimental rigor
4. **Conference format**: Switch to ACM by uncommenting line 9

## Adding Figures

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{carbon_comparison.png}
    \caption{Carbon emissions by inference strategy}
    \label{fig:carbon}
\end{figure}
```

## Target Venues

| Venue | Deadline | Format |
|-------|----------|--------|
| NeurIPS Workshop | May | neurips.cc |
| ICML | January | icml.cc |
| ACL (Green NLP) | February | aclweb.org |
| AAAI | August | aaai.org |

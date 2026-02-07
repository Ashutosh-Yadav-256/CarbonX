# CarbonX Paper - Publication Materials

## 📄 Files Created

### LaTeX Paper
- **File**: `carbonx_results.tex`
- **Type**: IEEE Conference format
- **Content**: Complete paper with methodology, results, and discussion

### Figures Script
- **File**: `generate_figures.py`
- **Generates**: 5 publication-quality figures (PNG + PDF)

---

## 🖼️ Figures Generated

1. **fig1_carbon_reduction.{png,pdf}** - Bar chart comparing baseline vs CarbonX emissions
2. **fig2_model_distribution.{png,pdf}** - Stacked bar showing model selection distribution
3. **fig3_accuracy_carbon.{png,pdf}** - Scatter plot of accuracy vs carbon trade-off
4. **fig4_aggregate_distribution.{png,pdf}** - Pie chart of aggregate model usage
5. **fig5_scale.{png,pdf}** - Carbon emissions at scale (100 to 1M queries)

---

## 🚀 How to Generate Figures

### Prerequisites:
```powershell
# Install matplotlib and seaborn (if not already installed)
.\.venv\Scripts\python.exe -m pip install matplotlib seaborn
```

### Generate:
```powershell
# Run the figure generation script
.\.venv\Scripts\python.exe paper\generate_figures.py
```

This creates all figures in `paper/figures/` directory.

---

## 📝 How to Compile LaTeX Paper

### Option 1: Overleaf (Recommended - No local setup needed)
1. Go to [https://www.overleaf.com](https://www.overleaf.com)
2. Create new project → "Upload Project"
3. Upload `carbonx_results.tex` and all figures from `paper/figures/`
4. Click "Recompile" to generate PDF

### Option 2: Local LaTeX
Requires LaTeX distribution (MiKTeX/TeX Live)

```powershell
# Navigate to paper directory
cd paper

# Compile (run twice for references)
pdflatex carbonx_results.tex
pdflatex carbonx_results.tex

# View PDF
carbonx_results.pdf
```

---

## 📊 Key Results in Paper

- **Primary Finding**: 66.7% carbon reduction across all benchmarks
- **Benchmarks**: 3,278 total questions (MMLU + GSM8K)
- **Model Distribution**: 87% small, 12% medium, 1% large
- **Language Accuracy**: 85.8-85.9% (excellent)
- **Carbon Savings**: 1.4 kg CO₂ per million queries

---

## ✏️ Customization

### Update Author Info:
Edit lines 12-15 in `carbonx_results.tex`:
```latex
\author{\IEEEauthorblockN{Your Name}
\IEEEauthorblockA{\textit{Your Institution}\\
Your City, Country \\
email@example.com}}
```

### Add More Figures:
Add `\includegraphics` commands in the appropriate sections:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\textwidth]{figures/figX_name.pdf}
\caption{Your caption here}
\label{fig:yourfig}
\end{figure}
```

---

## 📁 Directory Structure

```
paper/
├── carbonx_results.tex          # LaTeX paper
├── generate_figures.py          # Figure generation script
├── README.md                    # This file
└── figures/                     # Generated figures
    ├── fig1_carbon_reduction.{png,pdf}
    ├── fig2_model_distribution.{png,pdf}
    ├── fig3_accuracy_carbon.{png,pdf}
    ├── fig4_aggregate_distribution.{png,pdf}
    └── fig5_scale.{png,pdf}
```

---

## 🎯 Next Steps

1. **Generate figures**: Run `generate_figures.py`
2. **Review paper**: Read `carbonx_results.tex`
3. **Customize**: Add your name and affiliation
4. **Compile**: Use Overleaf or local LaTeX
5. **Submit**: To your target conference/journal!

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{carbonx2024,
  title={CarbonX: A Carbon-Aware Inference Framework for Sustainable Large Language Models},
  author={Your Name},
  booktitle={Conference Name},
  year={2024}
}
```

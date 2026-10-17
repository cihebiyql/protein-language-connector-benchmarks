# Recovered V3 ablation figures

This folder contains descriptive, reproducible figures from the independently revalidated **32 accepted runs out of 36 planned runs**. The four incomplete runs remain missing. No model loading, training, inference, new fitting, imputation, or significance testing was performed here.

- [Captioned figure report](report.pdf): four pages, including a short scope page.
- [Data health report](00_data_health.md): source hash, shape, missingness, duplicates, descriptive statistics and interpretation limits.
- [Captions](captions.md): figure definitions and conclusions.
- [Figure 1 — LM generation MAE](figures/png/fig_01_lm_mae.png).
- [Figure 2 — numeric-head MAE](figures/png/fig_02_numeric_mae.png).
- [Figure 3 — same-seed module comparisons](figures/png/fig_03_paired_module_deltas.png).
- [Process data](process_data/): CSV and XLSX tables. `01_planned_and_accepted_runs` retains all 36 planned positions and all displayed metrics; figure-specific tables expose every plotted point.

Individual symbols are the three predeclared seeds (17, 42, 2026). A mean and sample SD (`ddof=1`) are shown only when all three accepted seeds are available. Missing runs are never replaced with zero, and incomplete groups have no mean/SD summary. The paired plots use the same accepted seed on both sides; a positive difference means lower MAE for the candidate. LM and numeric endpoints stay separate. The resulting variation and sign changes do not establish stable improvement of the complete architecture. These figures make no claims about biological causation.

The analysis script expects the original `independent_audit.json` immediately above this directory. It verifies that the source is unchanged and that calculated summaries and module contrasts agree with that audit. From a compatible existing plotting environment:

```console
python build_figures.py
```

If `reportlab` is available in another existing Python environment, the report can be assembled independently, after plotting, without installing packages or changing either environment:

```console
python build_figures.py --pdf-only
```

The actual library versions used for plotting are in `figure_manifest.json`. The report was assembled with ReportLab 4.4.9. PDF, 300-dpi PNG and JPG exports are supplied. All three original PNG figures were visually inspected; the four-page report was rasterized and inspected for caption placement and clipping. Rasterized PDF previews are retained in `qa_preview/`.

# LLaVA-Med with Visual Contrastive Decoding

This repository is a fork of [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med), with additional scripts for chest X-ray report generation, Visual Contrastive Decoding (VCD), metric-based evaluation, parameter search, and plotting.

The upstream project documentation is preserved in [`README_llavamed.md`](README_llavamed.md). Refer to it for the original model, serving, and evaluation instructions. This README focuses on the added `PROJECT_*` workflow.

> **Research use only.** Follow the licenses and use restrictions of the upstream LLaVA-Med code, model, data, and all dependencies. Do not use generated reports or this software for clinical care or clinical decision-making.

## Contents

- [What was added](#what-was-added)
- [Requirements and setup](#requirements-and-setup)
- [Configure the data and model paths](#configure-the-data-and-model-paths)
- [Report-generation workflow](#report-generation-workflow)
- [VCD grid-search workflow](#vcd-grid-search-workflow)
- [Outputs](#outputs)
- [Known setup notes](#known-setup-notes)
- [Information to add](#information-to-add)
- [Acknowledgements](#acknowledgements)

## What was added

| File or folder | Purpose |
| --- | --- |
| `PROJECT_llavamed_vcd.py` | Generates chest X-ray reports with LLaVA-Med. In its current form, it generates both a baseline report and a VCD report for each selected study. |
| `PROJECT_grid_search.py` | Generates VCD reports for the `alpha`, `beta`, and `noise_step` combinations configured in the script. |
| `PROJECT_evaluation.py` | Compares baseline and VCD reports using RadGraph, RaTEScore, and CHAIR. Writes per-case scores and summary statistics. |
| `PROJECT_parameter_selection.py` | Scores grid-search reports with RadGraph, summarizes scores per parameter combination, and reports the best combination for a selected RadGraph metric. |
| `PROJECT_metrics.py` | A small, manually configured helper for computing RadGraph and RaTEScore for one results CSV. |
| `PROJECT_plot_grid.py` | Creates a RadGraph ER contour plot for a selected beta value from grid-search scores. |
| `PROJECT_plot_results.py` | Displays example comparison and difference plots using score arrays entered directly in the script. |
| `PROJECT_VCD/` | VCD noise and sampling implementation files. |
| `PROJECT_CHAIR/` | CHAIR/CheXbert files used by the evaluation pipeline. |

## Requirements and setup

The custom scripts depend on the upstream LLaVA-Med environment as well as metric and plotting packages. The model-generation scripts use PyTorch, Transformers, Pillow, and the LLaVA-Med code; evaluation uses RadGraph, RaTEScore, and CheXbert/CHAIR; plotting uses Matplotlib, NumPy, pandas, and SciPy.

1. Set up the upstream repository and environment following [`README_llavamed.md`](README_llavamed.md#install).
2. Install or otherwise make available the additional packages required by the scripts you intend to run.
3. Ensure the LLaVA-Med model checkpoint can be downloaded or accessed. The generation scripts default to `microsoft/llava-med-v1.5-mistral-7b`.
4. A CUDA-capable GPU is recommended for report generation and metric computation. CPU execution may be extremely slow.

> **Dependency versions:** [TODO: record the tested Python, PyTorch/CUDA, Transformers, RadGraph, RaTEScore, CheXbert, and plotting-package versions, ideally in a project-specific requirements file.]

## Configure the data and model paths

The generation scripts currently use hard-coded configuration values. Before running them, edit the relevant settings near the top of each script:

- `DATASET_ROOT` should point to the Indiana University chest X-ray dataset.
- The dataset root is expected to contain `indiana_reports.csv`, `indiana_projections.csv`, and `images/images_normalized/`.
- `N` controls how many usable studies are selected. In `PROJECT_llavamed_vcd.py`, `OPTIMIZATION_N` also controls the starting offset.
- In `PROJECT_grid_search.py`, update `ALPHA_VALUES`, `BETA_VALUES`, `NOISE_STEP_VALUES`, and `N` to define the grid and sample size.
- In `PROJECT_llavamed_vcd.py`, review `NOISE_STEP`, `CD_ALPHA`, `CD_BETA`, and `SEED`.
- Ensure the VCD implementation imports refer to the VCD folder present in this checkout; see [Known setup notes](#known-setup-notes).

The scripts expect the Indiana dataset columns used by their code, including report fields such as `uid`, `findings`, and `impression`, plus projection/image filename fields used to locate each image.

## Report-generation workflow

### Generate baseline and VCD reports

After configuring the data and VCD paths, run:

```bash
python PROJECT_llavamed_vcd.py
```

The script selects usable studies, loads the model, and writes a results CSV under `results/`. By default, the output name is `llavamed_results_450_vcd.csv` (the number reflects the configured sample size).

The generated CSV includes the source study information and report columns:

- `llavamed_report`: baseline report
- `llavamed_report_with_VCD`: VCD report

**Current behavior:** although the script has a top-level `USE_CD` setting, its inference loop is currently hard-coded to generate both baseline and VCD variants. Treat the paired output as the supported behavior unless the loop and corresponding output-column handling are changed together.

### Evaluate report quality

`PROJECT_evaluation.py` reads the paired reports, excludes rows missing any required reference or generated report, and evaluates the same remaining cases with:

- **RadGraph:** entity, entity-relation, and partial entity-relation scores
- **RaTEScore**
- **CHAIR:** CheXbert-based finding-label comparison

Example, with paths supplied explicitly:

```bash
python PROJECT_evaluation.py PROJECT_RESULTS/llavamed_results_450_vcd.csv --output PROJECT_RESULTS/llavamed_results_450_vcd_metrics.csv
```

The input must contain `findings`, `impression`, `llavamed_report`, and a VCD report column (`llavamed_report_with_VCD` or `llavamed_report_with_VDD`). The script writes per-case scores to the requested output and writes a second file with `_summary` appended to the output name, containing means and standard deviations.

Use `--vcd-column COLUMN_NAME` to select the VCD report column explicitly, or `--chexbert-checkpoint PATH` to provide a local CheXbert checkpoint. If no checkpoint is supplied, the script attempts to download the standard checkpoint.

## VCD grid-search workflow

### 1. Generate reports for the parameter grid

Configure the parameter values and dataset paths in `PROJECT_grid_search.py`, then run:

```bash
python PROJECT_grid_search.py
```

The default grid contains five values each for `alpha`, `beta`, and `noise_step` (125 combinations total), evaluated on 50 studies. That can require generating up to 6,250 reports and may take a long time. Start with a smaller `N` and smaller parameter lists to check the setup.

The default output is `llavamed_vcd_gridsearch_2_50.csv` in the repository root. It contains the parameter values, reference report fields, and generated report for each study/combination.

### 2. Score combinations and select a best setting

Pass the generated CSV to `PROJECT_parameter_selection.py`:

```bash
python PROJECT_parameter_selection.py llavamed_vcd_gridsearch_2_50.csv --output radgraph_grid_scores.csv --summary-output radgraph_grid_summary.csv --metric radgraph_er
```

This computes per-row RadGraph scores and averages them by `(alpha, beta, noise_step)`. The output files contain the per-row scores and the per-combination summary. The default selection metric is `radgraph_er`; the supported alternatives are `radgraph_e` and `radgraph_bar_er`.

Run `python PROJECT_parameter_selection.py --help` for all options. The input CSV must include the parameter columns, `findings`, `impression`, and `llavamed_report`.

### 3. Plot grid-search results

Use either the per-row score file or the parameter summary:

```bash
python PROJECT_plot_grid.py radgraph_grid_summary.csv --output PROJECT_RESULTS/radgraph_er_contour_beta_005.png
```

The plot script saves a contour image and marks the highest mean RadGraph ER point. It currently plots only `beta = 0.05`, configured by `TARGET_BETA` in `PROJECT_plot_grid.py`; change that value in the script to visualize another beta slice. Add `--show` to display the figure interactively as well.

## Outputs

Typical generated files include:

| Output | Produced by | Contents |
| --- | --- | --- |
| `PROJECT_RESULTS/llavamed_results_<N>_vcd.csv` | `PROJECT_llavamed_vcd.py` | Study data, reference reports, and paired baseline/VCD generated reports. |
| `llavamed_vcd_gridsearch_2_<N>.csv` | `PROJECT_grid_search.py` | Generated VCD reports and parameter values for each study and grid combination. |
| `radgraph_grid_scores.csv` | `PROJECT_parameter_selection.py` | Original grid rows with per-row RadGraph scores. |
| `radgraph_grid_summary.csv` | `PROJECT_parameter_selection.py` | Mean RadGraph scores and row counts grouped by parameter combination. |
| `PROJECT_RESULTS/llavamed_results_<N>_vcd_metrics_summary.csv` | `PROJECT_evaluation.py` | Mean and standard deviation of evaluation score columns. |
| `PROJECT_RESULTS/radgraph_er_contour_beta_005.png` | `PROJECT_plot_grid.py` | Contour visualization of mean RadGraph ER for the selected beta slice. |

Exact names may differ when script settings or command-line output paths are changed.

## Known setup notes

- The VCD imports and CHAIR subprocess path resolve to `PROJECT_VCD/` and `PROJECT_CHAIR/` relative to the script files, so they do not depend on the current working directory.
- `PROJECT_llavamed_vcd.py` runs both report-generation modes through its `using_vcd` loop. Changing only the top-level `USE_CD` value does not currently make this a baseline-only or VCD-only run.
- `PROJECT_metrics.py` uses the bundled `PROJECT_RESULTS/llavamed_results_450_vcd.csv` by default and builds its reference report from `findings` only. For the paired baseline/VCD comparison and full metric set, use `PROJECT_evaluation.py`.
- `PROJECT_plot_results.py` uses example values defined in arrays near the top of the file, displays two figures, and does not load evaluation CSVs or save the plots. Replace those values with your own results before using it for reporting.
- Output folders, dataset paths, checkpoint availability, and package/model access may need adjustment for your machine.

## Information to add

- [ ] Tested hardware, operating system, and environment setup
- [ ] Exact commands and parameter settings used for the final experiments
- [ ] Dataset source/version, selection criteria, and number of included studies
- [ ] Metric interpretation and the metric used to select the final parameters
- [ ] Final results and links to generated figures/tables
- [ ] Any additional licenses or acknowledgements applicable to the added code and data

## Acknowledgements

This repository builds on [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med). Please also cite and acknowledge the original LLaVA-Med and VCD work, the Indiana University chest X-ray dataset, and the metric implementations used in your experiments.

[TODO: Add the full citations and describe the authors' contributions to this fork.]

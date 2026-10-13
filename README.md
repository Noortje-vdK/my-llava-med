# LLaVA-Med with Visual Contrastive Decoding

This repository is a fork of [Microsoft LLaVA-Med](https://github.com/microsoft/LLaVA-Med), with additional scripts for chest X-ray report generation, Visual Contrastive Decoding (VCD), metric-based evaluation, parameter search, and plotting.

The upstream project documentation is preserved in [`README_llavamed.md`](README_llavamed.md). Refer to it for the original model, serving, and evaluation instructions. This README focuses on the added `PROJECT_*` workflow.

## What was added

| File or folder | Purpose |
| --- | --- |
| `PROJECT_llavamed_vcd.py` | Generates chest X-ray reports with LLaVA-Med. In its current form, it generates both a baseline report and a report with VCD for each selected study. |
| `PROJECT_grid_search.py` | Generates VCD reports for the `alpha`, `beta`, and `noise_step` combinations configured in the script. |
| `PROJECT_evaluation.py` | Compares baseline and VCD reports using RadGraph, RaTEScore, and CHAIR. Writes per-case scores and summary statistics. |
| `PROJECT_parameter_selection.py` | Scores grid-search reports with RadGraph, summarizes scores per parameter combination, and reports the best combination for a selected RadGraph metric. |
| `PROJECT_metrics.py` | A small helper for computing RadGraph and RaTEScore for one results CSV. |
| `PROJECT_plot_grid.py` | Creates a RadGraph ER contour plot for a selected beta value from grid-search scores. |
| `PROJECT_plot_results.py` | Displays example comparison and difference plots using score arrays entered directly in the script. |
| `PROJECT_VCD/` | VCD noise and sampling implementation files. |
| `PROJECT_CHAIR/` | CHAIR/CheXbert files used by the evaluation pipeline. |

## Requirements and setup

To run the code, it is suggested to use a GPU either from Google Colab (use `llava-med.ipynb` for this set-up)
or a sufficiently large GPU.

The following instructions are made to use with a notebook, where dependencies are handled inside the notebook.

> **Dependency versions:** [TODO: record the tested Python, PyTorch/CUDA, Transformers, RadGraph, RaTEScore, CheXbert, and plotting-package versions, ideally in a project-specific requirements file.]

## Configure the data and model paths

The generation scripts currently use hard-coded configuration values. Before running them, edit the relevant settings near the top of each script:

- `DATASET_ROOT` in **both** generation scripts should point to the Indiana University chest X-ray dataset. The scripts currently have different machine-specific paths.
- The dataset root is expected to contain `indiana_reports.csv`, `indiana_projections.csv`, and `images/images_normalized/`.
- `PROJECT_grid_search.py` takes the first `N` usable studies after filtering. `PROJECT_llavamed_vcd.py` skips the first `OPTIMIZATION_N` usable studies and then takes the next `N`. With the current settings, these are separate cohorts: studies 1–50 for grid search and studies 51–500 for report generation (assuming at least 500 usable studies and the same data/filter ordering).
- Update `ALPHA_VALUES`, `BETA_VALUES`, and `NOISE_STEP_VALUES` in `PROJECT_grid_search.py` to change the search grid.
- Update `NOISE_STEP`, `CD_ALPHA`, `CD_BETA`, and `SEED` in `PROJECT_llavamed_vcd.py` to change its fixed VCD settings.

### Current experiment settings

The values below are the settings currently present in the scripts. Keep this table in sync when changing the experiment configuration.

| Setting | `PROJECT_llavamed_vcd.py` | `PROJECT_grid_search.py` |
| --- | --- | --- |
| Model | `microsoft/llava-med-v1.5-mistral-7b` | `microsoft/llava-med-v1.5-mistral-7b` |
| VCD enabled | `USE_CD = True` | `USE_CD = True` |
| VCD contrast strength (`alpha`) | `CD_ALPHA = 1.25` | `[0.5, 0.75, 1.0, 1.25, 1.5]` |
| VCD cutoff (`beta`) | `CD_BETA = 0.05` | `[0.05, 0.1, 0.2, 0.3, 0.4]` |
| Diffusion noise step | `NOISE_STEP = 100` | `[100, 250, 500, 750, 900]` |
| Random seed | `SEED = 21` | `SEED = 20` |
| Number of studies (`N`) | `450` | `50` |
| Starting offset (`OPTIMIZATION_N`) | `50` | Not used |
| Decoding | Sampling enabled; temperature `0.7`; top-p `0.9`; max `512` new tokens | Sampling enabled; temperature `0.7`; top-p `0.9`; max `512` new tokens |
| Model loading | 4-bit enabled; 8-bit disabled | 4-bit enabled; 8-bit disabled |

The grid therefore contains `5 × 5 × 5 = 125` parameter combinations and generates up to `125 × 50 = 6,250` reports. `PROJECT_llavamed_vcd.py` generates both baseline and VCD reports for each of its 450 selected studies. Although `USE_CD` is `True`, the script's inference loop still explicitly runs both modes; do not interpret this flag as making that script VCD-only.

The dataset roots are currently set to `/home/20213144/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2` in `PROJECT_llavamed_vcd.py` and `/kaggle/input/chest-xrays-indiana-university` in `PROJECT_grid_search.py`. Replace each with the correct path for your environment.

### Evaluation and analysis defaults

| Script | Current default |
| --- | --- |
| `PROJECT_evaluation.py` | Input: `PROJECT_RESULTS/llavamed_results_450_vcd(1).csv`; output: corresponding `_metrics.csv`; VCD report column is auto-detected. |
| `PROJECT_parameter_selection.py` | Input: `merged.csv`; per-row output: `radgraph_grid_scores.csv`; grouped summary: `radgraph_grid_summary.csv`; batch size: `16`; best setting selected by `radgraph_er`. |
| `PROJECT_plot_grid.py` | Plots only `beta = 0.05`; default output: `PROJECT_RESULTS/radgraph_er_contour_beta_005.png`. |

The scripts expect the Indiana dataset columns used by their code, including report fields such as `uid`, `findings`, and `impression`, plus projection/image filename fields used to locate each image.

## Report-generation workflow

### Generate baseline and VCD reports

After configuring the data and VCD paths, run:

```bash
python PROJECT_llavamed_vcd.py
```

The script selects usable studies, loads the model, and writes a results CSV under `PROJECT_RESULTS/`. By default, the output name is `llavamed_results_450_vcd.csv` (the number reflects the configured sample size).

The generated CSV includes the source study information and report columns:

- `llavamed_report`: baseline report
- `llavamed_report_with_VCD`: VCD report

### Evaluate report quality

`PROJECT_evaluation.py` reads the paired reports, excludes rows missing any required reference or generated report, and evaluates the same remaining cases with:

- **RadGraph:** entity, entity-relation, and partial entity-relation scores
- **RaTEScore** extended RaTEScore with better synonym and negation handling
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
- `PROJECT_llavamed_vcd.py` runs both report-generation modes through its `using_vcd` loop. (Changing only the top-level `USE_CD` value does not currently make this a baseline-only or VCD-only run.)
- `PROJECT_metrics.py` uses the bundled `PROJECT_RESULTS/llavamed_results_450_vcd.csv` by default and builds its reference report from `findings` only. For the paired baseline/VCD comparison and full metric set, use `PROJECT_evaluation.py`.
- `PROJECT_plot_results.py` uses example values defined in arrays near the top of the file, displays two figures, and does not load evaluation CSVs or save the plots. Replace those values with your own results before using it for reporting.

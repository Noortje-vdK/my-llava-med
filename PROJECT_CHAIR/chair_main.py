"""
Author: GROUP 10
Last changed: 2026/09/27

Aim script: implementation of the CHAIR hallucination metric.
The implementation follows the implementation as done by "Med-VCD: Mitigating hallucination for medical large vision language models 
through visual contrastive decoding" (doi: https://doi.org/10.1016/j.compbiomed.2025.111347). Here, the evaluation metric based on MedHEval was used (https://github.com/Aofei-Chang/MedHEval).

In this script, gt refers to ground truth (so the original reports)

This script can be run after running the LLaVAMED script. Therefore, this script is for post-processing.
"""
# Import of some modules
import subprocess
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download # Necessary for import of CheXbert from Hugging Face

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent

# Directory of the results from Med-LLaVA
RESULTS_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else print("no output file yet to process")

# Folder for output files related to the CHAIR metric
CHAIR_DIR = OUTPUT_DIR / f"chair_{RESULTS_CSV.stem}"

# Importing CheXbert from Hugging Face; CHAIR uses CheXbert
CHEXBERT_LABEL_SCRIPT = OUTPUT_DIR / "chair_utils" / "chexbert" / "label.py"
CHEXBERT_PATH = hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth")

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"

# Some functions related to the computation of the metrics
def calculate_metrics(inference_csv, ground_truth_csv):
    """
    Docstring for calculate_metrics

    This function is copied from MedHeval (https://github.com/Aofei-Chang/MedHEval), specifically from code/evaluation/report_eval/CXRMetric/run_extraction.py.

    Aim of script: compare labels of each generated report with those of its reference report. 
    Labels are based on ChexBert. The labels / findings are:  Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation,
    Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding.
    """
    # Load CSV files
    df_infer = pd.read_csv(inference_csv)
    df_gt = pd.read_csv(ground_truth_csv)

    # Ensure both CSVs have the same number of rows
    assert len(df_infer) == len(df_gt), "The two CSV files must have the same number of rows"

    # Get symptom columns (excluding 'report' column)
    symptom_columns = [col for col in df_infer.columns if col != "report"]

    hallucinated_count = 0
    total_generated_count = 0
    true_positive_count = 0
    false_negative_count = 0
    true_negative_count = 0
    false_positive_count = 0
    total_positive_gt = 0

    for symptom in symptom_columns:
        # Get the inferred and ground truth labels for the current symptom
        infer_labels = df_infer[symptom].fillna("")
        gt_labels = df_gt[symptom].fillna("")

        # Convert non-empty values to numeric (-1, 0, 1) and ignore empty values
        infer_labels = pd.to_numeric(infer_labels, errors='coerce')
        gt_labels = pd.to_numeric(gt_labels, errors='coerce')

        # Filter out cases where ground truth is uncertain (-1)
        valid_indices = gt_labels != -1
        infer_labels = infer_labels[valid_indices]
        gt_labels = gt_labels[valid_indices]

        # Compute metrics
        tp = ((infer_labels == 1) & (gt_labels == 1)).sum()
        tn = ((infer_labels == 0) & (gt_labels == 0)).sum()
        fp = ((infer_labels == 1) & (gt_labels == 0)).sum()
        fn = ((infer_labels == 0) & (gt_labels == 1)).sum()

        hallucinated_count += fp + fn
        total_generated_count += ((infer_labels != -1) & (infer_labels.notna())).sum()
        true_positive_count += tp
        false_negative_count += fn
        true_negative_count += tn
        false_positive_count += fp
        total_positive_gt += (gt_labels == 1).sum()

    # Calculate hallucination score
    hallucination_score = hallucinated_count / total_generated_count if total_generated_count > 0 else 0

    # Calculate recall (sensitivity)
    recall = true_positive_count / total_positive_gt if total_positive_gt > 0 else 0

    # Calculate specificity
    specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0

    # Calculate accuracy
    accuracy = (true_positive_count + true_negative_count) / (true_positive_count + true_negative_count + false_positive_count + false_negative_count) if (true_positive_count + true_negative_count + false_positive_count + false_negative_count) > 0 else 0

    return hallucination_score, recall, specificity, accuracy

def calculate_article_chair(inference_csv, ground_truth_csv):
    """
    Docstring for calculate_article_chair

    This function calculates the evaluation metrics, as mentioned in https://doi.org/10.1016/j.compbiomed.2025.111347.

    Two scores are given as output, namely:
    1) CHAIR = (FP + FN) / (# generated findings labelled 1 or 0)
    2) Recall = TP / (# findings labelled 1 in the reference)

    These labels are given as output by the CheXbert.
    """
    # Selecting results with label 1
    generated = pd.read_csv(inference_csv).drop(columns="report") == 1
    reference = pd.read_csv(ground_truth_csv).drop(columns="report") == 1

    # Determining total number of results with label 1 
    n_generated = generated.values.sum()
    n_reference = reference.values.sum()

    # Calculating metrics following definitions as stated above
    chair = (generated & ~reference).values.sum() / n_generated if n_generated > 0 else 0
    recall = (generated & reference).values.sum() / n_reference if n_reference > 0 else 0
    return chair, recall


def label_reports(csv_path: Path, out_dir: Path) -> Path:
    """
    Docstring labels_report

    Follows the calc_metric function in MedHEval.
    Aim: extracting the labels generated by CheXBert

    Subprocess is done following the architecture of MedHEval
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(CHEXBERT_LABEL_SCRIPT),
         "-c", CHEXBERT_PATH, "-d", str(csv_path), "-o", str(out_dir)],
        check=True)
    return out_dir / "labeled_reports.csv"

# The code below follows the architecture of run_chair.py (https://github.com/Aofei-Chang/MedHEval)

# Loading generated reports
results = pd.read_csv(RESULTS_CSV).fillna("")

# Only remain the reports where LLaVAMED actually returns an output
results = results[~results["llavamed_report"].str.startswith("ERROR processing")]

# Turning the files into CSV format
gt = pd.DataFrame({
    STUDY_ID_COL_NAME: results["uid"],
    REPORT_COL_NAME: (results["findings"] + " " + results["impression"]).str.strip()})
pred = pd.DataFrame({
    STUDY_ID_COL_NAME: results["uid"],
    REPORT_COL_NAME: results["llavamed_report"].str.strip()})

# MedHeVAL: in case the length of reports is less then 10 tokens, report is set as empty.
pred.loc[pred[REPORT_COL_NAME].str.len() < 10, REPORT_COL_NAME] = "No report generated."

# Turning the files into CSV format
CHAIR_DIR.mkdir(parents=True, exist_ok=True)
gt_csv = CHAIR_DIR / "gt_reports.csv"
pred_csv = CHAIR_DIR / "pred_reports.csv"
gt.to_csv(gt_csv, index=False)
pred.to_csv(pred_csv, index=False)

# Extracting the labels generated with CHeXBert.
pred_label_file = label_reports(pred_csv, CHAIR_DIR / "pred_labels")
gt_label_file = label_reports(gt_csv, CHAIR_DIR / "gt_labels")

# Computing all metrics
# Mind: the original CHAIR is different from the CHAIR as mentioned by https://doi.org/10.1016/j.compbiomed.2025.111347.
# The latter is reffered to as tthe CHAIR and recall from the article.
hallucination_score, recall, specificity, accuracy = calculate_metrics(pred_label_file, gt_label_file)
article_chair, article_recall = calculate_article_chair(pred_label_file, gt_label_file)

# Saving results, output is a .txt file containing all metric scores
out_file = CHAIR_DIR / "eval_chair_res.txt"
with open(out_file, "w") as f:
    f.write("CHAIR: " + str(hallucination_score) + "\n")
    f.write("Recall: " + str(recall) + "\n")
    f.write("Specificity: " + str(specificity) + "\n")
    f.write("Accuracy: " + str(accuracy) + "\n")
    f.write("CHAIR (article formula): " + str(article_chair) + "\n")
    f.write("Recall (article formula): " + str(article_recall) + "\n")

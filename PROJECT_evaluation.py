import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
#file path of the reports file you want to evaluate
file_path = "PROJECT_RESULTS/llavamed_results_450_vcd(1).csv" 

BASE_REPORT_COLUMN = "llavamed_report"
VCD_REPORT_COLUMNS = ("llavamed_report_with_VCD", "llavamed_report_with_VDD")
REFERENCE_COLUMNS = ("findings", "impression")
RADGRAPH_METRICS = ("e", "er", "bar_er")
CHAIR_SCRIPT = (
    Path(__file__).resolve().parent
    / "PROJECT_CHAIR"
    / "chair_utils"
    / "chexbert"
    / "label.py"
)


def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def make_reference(row):
    findings = clean_text(row["findings"])
    impression = clean_text(row["impression"])
    return f"Findings: {findings}\nImpression: {impression}"


def get_vcd_column(df, requested_column):
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(f"VCD report column {requested_column!r} was not found")
        return requested_column

    for column in VCD_REPORT_COLUMNS:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a VCD report column. Expected one of: "
        + ", ".join(VCD_REPORT_COLUMNS)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score base and VCD-generated reports with RadGraph, RaTEScore, and CHAIR."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=file_path,
        help="Input results CSV.",
    )
    parser.add_argument(
        "--output",
        default=file_path.replace(".csv", "_metrics.csv"), 
        help="Output CSV containing per-row scores.",
    )
    parser.add_argument(
        "--vcd-column",
        help="VCD report column; auto-detects the VCD or VDD spelling by default.",
    )
    parser.add_argument(
        "--chexbert-checkpoint",
        help="Path to a CheXbert checkpoint; downloads the standard checkpoint if omitted.",
    )
    return parser.parse_args()


def normalize_radgraph_reward(reward):
    if (
        len(reward) == 1
        and isinstance(reward[0], (list, tuple))
        and len(reward[0]) == len(RADGRAPH_METRICS)
    ):
        reward = reward[0]

    if len(reward) != len(RADGRAPH_METRICS):
        raise RuntimeError(
            f"Expected {len(RADGRAPH_METRICS)} RadGraph components, got {reward!r}"
        )

    scores = []
    for score in reward:
        if isinstance(score, (list, tuple)):
            if len(score) != 1:
                raise RuntimeError(f"Unexpected RadGraph score format: {score!r}")
            score = score[0]
        scores.append(float(score))
    return scores


def score_radgraph(references, hypotheses):
    from radgraph import F1RadGraph

    scorer = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    scores = []
    for index, (reference, hypothesis) in enumerate(zip(references, hypotheses), start=1):
        _, reward, _, _ = scorer(hyps=[hypothesis], refs=[reference])
        if len(reward) != len(RADGRAPH_METRICS):
            raise RuntimeError(f"Unexpected RadGraph reward at row {index}: {reward!r}")
        scores.append(normalize_radgraph_reward(reward))
        if index % 25 == 0 or index == len(references):
            print(f"RadGraph: scored {index}/{len(references)} reports")
    return scores


def score_ratescore(references, hypotheses):
    from RaTEScore import RaTEScore

    scores = RaTEScore().compute_score(hypotheses, references)
    if len(scores) != len(references):
        raise RuntimeError(
            f"RaTEScore returned {len(scores)} scores for {len(references)} reports"
        )
    return [float(score) for score in scores]


def get_chexbert_checkpoint(checkpoint_path):
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CheXbert checkpoint not found: {path}")
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth"))


def label_all_reports(references, base_hypotheses, vcd_hypotheses, checkpoint_path):
    if not CHAIR_SCRIPT.is_file():
        raise FileNotFoundError(f"CHAIR labeling script not found: {CHAIR_SCRIPT}")

    report_count = len(references)
    all_reports = references + base_hypotheses + vcd_hypotheses

    with tempfile.TemporaryDirectory(prefix="vcd_chair_") as temp_dir:
        temp_path = Path(temp_dir)
        reports_path = temp_path / "reports.csv"
        output_dir = temp_path / "labels"
        pd.DataFrame({"report": all_reports}).to_csv(reports_path, index=False)

        # subprocess.run(
        #     [
        #         sys.executable,
        #         str(CHAIR_SCRIPT),
        #         "--checkpoint",
        #         str(checkpoint_path),
        #         "--data",
        #         str(reports_path),
        #         "--output_dir",
        #         str(output_dir),
        #     ],
        #     check=True,
        #     cwd=CHAIR_SCRIPT.parent,
        # )

        # labeled = pd.read_csv(output_dir / "labeled_reports.csv")
        result = subprocess.run(
            [
                sys.executable,
                str(CHAIR_SCRIPT),
                "--checkpoint",
                str(checkpoint_path),
                "--data",
                str(reports_path),
                "--output_dir",
                str(output_dir),
            ],
            cwd=CHAIR_SCRIPT.parent,
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)

        print("STDERR:")
        print(result.stderr)

        result.check_returncode()

        labeled = pd.read_csv(output_dir / "labeled_reports.csv")

        if len(labeled) != 3 * report_count:
            raise RuntimeError(
                f"CHAIR labeled {len(labeled)} reports; expected {3 * report_count}"
            )

        return (
            labeled.iloc[:report_count].reset_index(drop=True),
            labeled.iloc[report_count : 2 * report_count].reset_index(drop=True),
            labeled.iloc[2 * report_count :].reset_index(drop=True),
        )


def calculate_chair_scores(reference_labels, hypothesis_labels):
    finding_columns = [column for column in reference_labels.columns if column != "report"]
    if not finding_columns:
        raise RuntimeError("CHAIR label files contain no finding columns")

    reference = reference_labels[finding_columns].apply(pd.to_numeric, errors="coerce")
    hypothesis = hypothesis_labels[finding_columns].apply(pd.to_numeric, errors="coerce")

    valid = reference != -1
    ref_valid = reference.where(valid)
    hyp_valid = hypothesis.where(valid)

    true_positive = ((hyp_valid == 1) & (ref_valid == 1)).to_numpy().sum()
    true_negative = ((hyp_valid == 0) & (ref_valid == 0)).to_numpy().sum()
    false_positive = ((hyp_valid == 1) & (ref_valid == 0)).to_numpy().sum()
    false_negative = ((hyp_valid == 0) & (ref_valid == 1)).to_numpy().sum()

    total_generated = ((hypothesis != -1) & hypothesis.notna()).to_numpy().sum()
    total_reference_positive = (reference == 1).to_numpy().sum()
    standard_total = true_positive + true_negative + false_positive + false_negative

    generated_positive = hypothesis == 1
    reference_positive = reference == 1
    generated_positive_count = generated_positive.to_numpy().sum()
    article_true_positive = (generated_positive & reference_positive).to_numpy().sum()

    return {
        "chair_hallucination_score": (
            (false_positive + false_negative) / total_generated if total_generated else 0.0
        ),
        "chair_recall": (
            true_positive / total_reference_positive if total_reference_positive else 0.0
        ),
        "chair_specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        ),
        "chair_accuracy": (true_positive + true_negative) / standard_total if standard_total else 0.0,
        "chair_article": (
            (generated_positive_count - article_true_positive) / generated_positive_count
            if generated_positive_count
            else 0.0
        ),
        "chair_article_recall": (
            article_true_positive / total_reference_positive if total_reference_positive else 0.0
        ),
    }


def add_scores(df, label, radgraph_scores, ratescore_scores, chair_scores):
    for metric_index, metric in enumerate(RADGRAPH_METRICS):
        df[f"radgraph_{metric}_{label}"] = [score[metric_index] for score in radgraph_scores]
    df[f"ratescore_{label}"] = ratescore_scores
    for metric, scores in chair_scores.items():
        df[f"{metric}_{label}"] = scores


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    df_original = pd.read_csv(input_path)

    required_columns = {"findings", "impression", BASE_REPORT_COLUMN}
    missing = required_columns.difference(df_original.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
    if df_original.empty:
        raise ValueError("Input CSV contains no rows")

    vcd_column = get_vcd_column(df_original, args.vcd_column)

    # ------------------------------------------------------------------
    # Build a single clean evaluation cohort.
    #
    # A case is included only when all four components required for the
    # evaluation are present and non-empty:
    #   1. findings
    #   2. impression
    #   3. base LLaVA-Med report
    #   4. VCD/VDD LLaVA-Med report
    #
    # The same filtered dataframe is then used for RadGraph, RaTEScore,
    # and CHAIR, ensuring that all metrics are calculated on exactly the
    # same cases.
    # ------------------------------------------------------------------
    required_for_evaluation = [
        "findings",
        "impression",
        BASE_REPORT_COLUMN,
        vcd_column,
    ]

    valid_mask = pd.Series(True, index=df_original.index)
    exclusion_reasons = {}

    for row_index, row in df_original.iterrows():
        reasons = []

        for column in required_for_evaluation:
            value = row[column]

            if pd.isna(value):
                reasons.append(f"{column} is missing")
            elif not str(value).strip():
                reasons.append(f"{column} is empty")

        if reasons:
            exclusion_reasons[row_index] = reasons
            valid_mask.loc[row_index] = False

    df = df_original.loc[valid_mask].copy().reset_index(drop=True)

    # report the filtering results
    original_count = len(df_original)
    included_count = len(df)
    excluded_count = original_count - included_count

    print("\n" + "=" * 72)
    print("EVALUATION COHORT")
    print("=" * 72)
    print(f"Input file:       {input_path}")
    print(f"VCD report column: {vcd_column}")
    print(f"Total input cases: {original_count}")
    print(f"Included cases:    {included_count}")
    print(f"Excluded cases:    {excluded_count}")

    if exclusion_reasons:
        print("\nExcluded cases:")
        for row_index, reasons in exclusion_reasons.items():
            row = df_original.loc[row_index]

            if "uid" in df_original.columns:
                uid = row["uid"]
            else:
                uid = row_index

            filename = row["filename"] if "filename" in df_original.columns else "N/A"

            print(
                f"  uid={uid} | filename={filename} | "
                f"reason: {'; '.join(reasons)}"
            )
    else:
        print("\nNo incomplete cases were excluded.")

    if df.empty:
        raise ValueError(
            "No complete cases remain after cohort filtering. "
            "Nothing can be evaluated."
        )

    print("=" * 72 + "\n")

    # report the number of cases that will be scored
    references = df.apply(make_reference, axis=1).tolist()
    base_hypotheses = df[BASE_REPORT_COLUMN].map(clean_text).tolist()
    vcd_hypotheses = df[vcd_column].map(clean_text).tolist()

    # Defensive checks: these should never fail after cohort filtering.
    if not (len(references) == len(base_hypotheses) == len(vcd_hypotheses)):
        raise RuntimeError(
            "Internal cohort alignment error: references and hypotheses "
            "have different lengths."
        )

    print(f"Scoring {len(df)} complete cases.\n")

    # radgraph
    print("Scoring llavamed_report with RadGraph...")
    base_radgraph = score_radgraph(references, base_hypotheses)

    print("Scoring VCD reports with RadGraph...")
    vcd_radgraph = score_radgraph(references, vcd_hypotheses)

    # ratescore
    print("Scoring llavamed_report with RaTEScore...")
    base_ratescore = score_ratescore(references, base_hypotheses)

    print("Scoring VCD reports with RaTEScore...")
    vcd_ratescore = score_ratescore(references, vcd_hypotheses)

    # chair
    print("\nLoading CheXbert checkpoint and running CHAIR labels...")
    checkpoint = get_chexbert_checkpoint(args.chexbert_checkpoint)

    reference_labels, base_labels, vcd_labels = label_all_reports(
        references,
        base_hypotheses,
        vcd_hypotheses,
        checkpoint,
    )

    base_chair = calculate_chair_scores(reference_labels, base_labels)
    vcd_chair = calculate_chair_scores(reference_labels, vcd_labels)

    # add scores to the dataframe
    add_scores(df, "llavamed", base_radgraph, base_ratescore, base_chair)
    add_scores(df, "vcd", vcd_radgraph, vcd_ratescore, vcd_chair)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved per-row scores to: {output_path}")

    # per row summary
    print("\nPer-row summary:")
    for row_index, row in df.iterrows():
        uid = row["uid"] if "uid" in df.columns else row_index
        print(
            f"uid={uid}: "
            f"RadGraph llava={row['radgraph_bar_er_llavamed']:.4f}, "
            f"VCD={row['radgraph_bar_er_vcd']:.4f}; "
            f"RaTEScore llava={row['ratescore_llavamed']:.4f}, "
            f"VCD={row['ratescore_vcd']:.4f}; "
            f"CHAIR llava={row['chair_article_llavamed']:.4f}, "
            f"VCD={row['chair_article_vcd']:.4f}"
        )


    # mean and std 
    score_columns = [
        column
        for column in df.columns
        if column.startswith(("radgraph_", "ratescore_", "chair_"))
    ]

    summary_stats = df[score_columns].agg(["mean", "std"]).T
    summary_stats.columns = ["mean", "std"]

    print("\nMean ± Standard Deviation:")
    print(summary_stats.to_string(float_format=lambda x: f"{x:.4f}"))

    # Save summary statistics to a separate CSV file
    summary_path = output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )
    summary_stats.to_csv(summary_path)

    print(f"\nSaved mean and standard deviation to: {summary_path}")


    # final summary
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Input cases:      {original_count}")
    print(f"Evaluated cases:  {included_count}")
    print(f"Excluded cases:   {excluded_count}")

    if exclusion_reasons:
        print("\nExclusion breakdown:")
        reason_counts = {}

        for reasons in exclusion_reasons.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in sorted(reason_counts.items()):
            print(f"  {count:>3} × {reason}")

    print("=" * 72)


if __name__ == "__main__":
    main()
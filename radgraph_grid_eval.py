import argparse
import re

import pandas as pd
from radgraph import F1RadGraph


# Columns that must exist in the input CSV
REQUIRED_COLUMNS = {
    "alpha",
    "beta",
    "noise_step",
    "findings",
    "impression",
    "llavamed_report",
}

# Parameter columns used to define a unique grid-search combination
PARAMETER_COLUMNS = ["alpha", "beta", "noise_step"]

# RadGraph metrics returned by F1RadGraph(reward_level="all")
REWARD_COLUMNS = ["radgraph_e", "radgraph_er", "radgraph_bar_er"]


def clean_text(value):
    """Clean NaN values and normalize whitespace."""
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def make_reference(row):
    """
    Combine the reference findings and impression into one report.
    """
    findings = clean_text(row["findings"])
    impression = clean_text(row["impression"])

    return f"Findings: {findings}\nImpression: {impression}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-row RadGraph scores and average them "
            "for each grid-search parameter combination."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="merged.csv",
        help="Input CSV file.",
    )

    parser.add_argument(
        "--output",
        default="radgraph_grid_scores.csv",
        help="Output CSV containing per-row RadGraph scores.",
    )

    parser.add_argument(
        "--summary-output",
        default="radgraph_grid_summary.csv",
        help="Output CSV containing average scores per parameter combination.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of rows to process in each progress batch.",
    )

    parser.add_argument(
        "--metric",
        choices=REWARD_COLUMNS,
        default="radgraph_bar_er",
        help=(
            "RadGraph metric used to determine the best parameter "
            "combination. Default: radgraph_bar_er"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    # ---------------------------------------------------------
    # Load input CSV
    # ---------------------------------------------------------
    df = pd.read_csv(args.input)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Input CSV contains no rows.")

    # ---------------------------------------------------------
    # Create reference and hypothesis reports
    # ---------------------------------------------------------
    references = df.apply(make_reference, axis=1).tolist()
    hypotheses = df["llavamed_report"].map(clean_text).tolist()

    # Initialize score columns
    for column in REWARD_COLUMNS:
        df[column] = pd.NA

    # ---------------------------------------------------------
    # Initialize RadGraph
    # ---------------------------------------------------------
    print("Loading RadGraph model...")

    radgraph = F1RadGraph(
        reward_level="all",
        model_type="radgraph-xl",
    )

    print("RadGraph model loaded.\n")

    # ---------------------------------------------------------
    # Calculate per-row RadGraph scores
    # ---------------------------------------------------------
    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))

        print(f"Processing rows {start}–{end - 1}...")

        # RadGraph's F1 API returns aggregate scores for a batch.
        # We process each reference/hypothesis pair individually so
        # that we retain a score for every row.
        for row_index in range(start, end):

            _, reward_list, _, _ = radgraph(
                hyps=[hypotheses[row_index]],
                refs=[references[row_index]],
            )

            if len(reward_list) != 3:
                raise RuntimeError(
                    f"Expected 3 RadGraph scores, "
                    f"got {len(reward_list)}: {reward_list!r}"
                )

            # RadGraph may return each score wrapped in a list.
            # Extract the scalar value for each metric.
            scores = []

            for score in reward_list:
                if isinstance(score, (list, tuple)):
                    if len(score) != 1:
                        raise RuntimeError(
                            f"Unexpected RadGraph score format: {score!r}"
                        )
                    score = score[0]

                scores.append(float(score))

            radgraph_e = scores[0]
            radgraph_er = scores[1]
            radgraph_bar_er = scores[2]

            df.at[row_index, "radgraph_e"] = radgraph_e
            df.at[row_index, "radgraph_er"] = radgraph_er
            df.at[row_index, "radgraph_bar_er"] = radgraph_bar_er

            print(
                f"  row {row_index}: "
                f"e={radgraph_e:.4f}, "
                f"er={radgraph_er:.4f}, "
                f"bar_er={radgraph_bar_er:.4f}"
            )

    # ---------------------------------------------------------
    # Convert score columns to numeric
    # ---------------------------------------------------------
    for column in REWARD_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # Save per-row scores
    # ---------------------------------------------------------
    df.to_csv(args.output, index=False)

    print(
        f"\nSaved per-row RadGraph scores to: "
        f"{args.output}"
    )

    # ---------------------------------------------------------
    # Calculate average score for each parameter combination
    # ---------------------------------------------------------
    summary_df = (
        df.groupby(PARAMETER_COLUMNS, dropna=False)[REWARD_COLUMNS]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Flatten the multi-level column names produced by .agg()
    summary_df.columns = [
        "_".join(column).strip("_")
        if isinstance(column, tuple)
        else column
        for column in summary_df.columns
    ]

    # Rename count columns to make their meaning explicit
    summary_df = summary_df.rename(
        columns={
            "radgraph_e_count": "num_rows",
        }
    )

    # ---------------------------------------------------------
    # Sort combinations by the selected metric
    # ---------------------------------------------------------
    selected_mean_column = f"{args.metric}_mean"

    summary_df = summary_df.sort_values(
        by=selected_mean_column,
        ascending=False,
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------
    summary_df.to_csv(
        args.summary_output,
        index=False,
    )

    print(
        f"Saved parameter-combination averages to: "
        f"{args.summary_output}"
    )

    # ---------------------------------------------------------
    # Print all parameter combinations
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("AVERAGE RADGRAPH SCORE PER PARAMETER COMBINATION")
    print("=" * 80)

    for _, row in summary_df.iterrows():

        print(
            f"alpha={row['alpha']}, "
            f"beta={row['beta']}, "
            f"noise_step={row['noise_step']} | "
            f"n={int(row['num_rows'])} | "
            f"e={row['radgraph_e_mean']:.4f} | "
            f"er={row['radgraph_er_mean']:.4f} | "
            f"bar_er={row['radgraph_bar_er_mean']:.4f}"
        )

    # ---------------------------------------------------------
    # Find and print best parameter combination
    # ---------------------------------------------------------
    best_row = summary_df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST PARAMETER COMBINATION")
    print("=" * 80)

    print(
        f"Metric used: {args.metric}"
    )

    print(
        f"alpha      = {best_row['alpha']}"
    )

    print(
        f"beta       = {best_row['beta']}"
    )

    print(
        f"noise_step = {best_row['noise_step']}"
    )

    print(
        f"Average radgraph_e      = "
        f"{best_row['radgraph_e_mean']:.4f}"
    )

    print(
        f"Average radgraph_er     = "
        f"{best_row['radgraph_er_mean']:.4f}"
    )

    print(
        f"Average radgraph_bar_er = "
        f"{best_row['radgraph_bar_er_mean']:.4f}"
    )

    print(
        f"Number of rows           = "
        f"{int(best_row['num_rows'])}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()

import re
import pandas as pd
from radgraph import F1RadGraph
from RaTEScore import RaTEScore

def load_reports(path):
    """
    Input: csv-file with provided and generated information for each image.
    Formulates a reference report with the combined Findings and Impressions of the reference
    and returns the reference and generated reports as lists of strings
    Output:
    - df: the altered dataframe.
    - refs: list of strings of reference reports.
    - gens: list of strings of generated reports.
    """

    # load and clean data
    df = pd.read_csv(path)
    df_clean = clean_df(df)

    # create reference report
    df["reference_report"] = df.apply(
        lambda row: f"Findings: {row['findings']}", # Impressions: {row['impression']}", --> nu alleen findings
        axis=1
    )

    # clean final reports
    df["reference_report_clean"] = df["reference_report"].apply(normalize_reports)
    df["generated_report_clean"] = df["llavamed_report"].apply(normalize_reports)

    # return as lists of strings
    refs = df["reference_report_clean"].astype(str).tolist()
    gens = df["generated_report_clean"].astype(str).tolist()

    return df, refs, gens

        
def normalize_reports(report):
    """
    Input: each report is inputted as a separate string

    Output: the function returns a "cleaned" string that reduces differences
    caused by formatting of the text:
    - removal of "Findings" header
    - removal of impression section
    - removal of bullet points
    - removal of enters and tabs
    - trailing whitespace
    """

    report = re.sub(r"^\s*[\*\-\•]\s*", "", report, flags=re.MULTILINE)
    report = re.sub(r"\s+", " ", report)
    report = report.strip()

    return report

def clean_df(df):
    """
    Input: raw dataframe
    Output: pre-processed dataframe that is clean to use
    """
    df_clean = df
    # assume missing findings and impressions mean there is nothing to comment
    df_clean["findings"] = df_clean["findings"].fillna("").astype(str)
    df_clean["impression"] = df_clean["impression"].fillna("").astype(str)
    df_clean["llavamed_report"] = df_clean["llavamed_report"].fillna("").astype(str)

    return df_clean

data, refs, hyps = load_reports("llavamed_results_10.csv")
print("PROCESSED DATA")
print(data.head())
assert len(refs) == len(hyps) # check if every reference report is linked to a generated report

# implementation of RadGraph F1
f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)

rg_e, rg_er, rg_bar_er = mean_reward

print("RADGRAPH METRIC")
print(mean_reward)

# implementation of RaTEScore metric
ratescore = RaTEScore()
scores = ratescore.compute_score(hyps, refs)

print("RATESCORE METRIC")
print(scores)
print("average RaTEScore")
print(sum(scores)/len(scores))
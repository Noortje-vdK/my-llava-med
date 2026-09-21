import re
import json

from radgraph import F1RadGraph
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

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
    report = re.sub(r"\bfindings\s*:?", "", report, flags=re.IGNORECASE)
    report = re.sub(r"\bimpression\b.*", "", report, flags=re.IGNORECASE | re.DOTALL) # removes entire impression section because it is cut off for now.
    report = re.sub(r"\bim\b", "", report, flags=re.IGNORECASE) # removes the "Im" that is cut off in this example
    report = re.sub(r"^\s*[\*\-\•]\s*", "", report, flags=re.MULTILINE)
    report = re.sub(r"\s+", " ", report)
    report = report.strip()

    return report

# load reports
with open("reference_reports.json", "r") as f:
    reference_reports = json.load(f)

with open("generated_reports.json", "r") as f:
    generated_reports = json.load(f)

# preprocess reports
selected_ref_report = reference_reports["5"] # select report using uid - manually selected here
norm_sel_ref_report = normalize_reports(selected_ref_report)

selected_gen_report = generated_reports["5"] # select report using uid - manually selected here
norm_sel_gen_report = normalize_reports(selected_gen_report)

"""
print("REFERENCE REPORT PRE")
print(selected_ref_report)
print("REFERENCE REPORT POST")
print(norm_sel_ref_report)

print("GENERATED REPORT PRE")
print(selected_gen_report)
print("GENERATED REPORT POST")
print(norm_sel_gen_report)
"""

# reports saved as a list of strings
refs = [norm_sel_ref_report]
hyps = [norm_sel_gen_report]

print(refs)
print(hyps)

# implementation of RadGraph F1
f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)

rg_e, rg_er, rg_bar_er = mean_reward

print("RADGRAPH METRIC")
print(mean_reward)

"""
# LLM-as-a-jugde hallucination metric
test_case = LLMTestCase(
    input="You are an expert radiologist. Write a concise radiology report for this frontal chest X-ray. Use Findings and Impression sections. Describe only findings supported by the image and state uncertainty when necessary.",
    actual_output=norm_sel_gen_report,
    context=[norm_sel_ref_report]
)
metric = HallucinationMetric(threshold=0.5)

print("LLM-as-a-judge HALLUCINATION MODEL")
evaluate(test_cases=[test_case], metrics=[metric])
"""

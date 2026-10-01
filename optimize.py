from pathlib import Path
from typing import Any
import pandas as pd
import os
import sys
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from transformers import MistralForCausalLM, set_seed


#for vcd
VCD_PATH = "/content/my-llava-med/VCD"

if VCD_PATH not in sys.path:
    sys.path.insert(0, VCD_PATH)

from VCD.vcd_add_noise import add_diffusion_noise
from VCD.vcd_sample import evolve_vcd_sampling


#settings
USE_CD = True

SEED = 20

#grid to search
ALPHA_VALUES = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
]

BETA_VALUES = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
]

NOISE_STEP_VALUES = [
    100,
    250,
    500,
    750,
    900,
]

#number of studies to search on
N = 50


#dataset
DATASET_ROOT = Path(
    r"/kaggle/input/chest-xrays-indiana-university"
)

REPORTS_CSV = DATASET_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV = DATASET_ROOT / "indiana_projections.csv"
IMAGE_DIR = DATASET_ROOT / "images" / "images_normalized"


#model
MODEL_NAME = "microsoft/llava-med-v1.5-mistral-7b"

OUTPUT_DIR = Path(__file__).resolve().parent

OUTPUT_CSV = (
    OUTPUT_DIR
    / f"llavamed_vcd_gridsearch_{N}.csv"
)


# device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "CUDA device:",
        torch.cuda.get_device_name(0)
    )
else:
    print(
        "CUDA is not available. "
        "LLaVA-MED 7B will be very slow."
    )

if USE_CD:
    evolve_vcd_sampling()

gc.collect()
torch.cuda.empty_cache()


print("\nChecking data files")

required_files = [
    REPORTS_CSV,
    PROJECTIONS_CSV,
    IMAGE_DIR,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}"
        )

print("Dataset paths OK.")


#load data
print("\nLoading dataset")

reports = pd.read_csv(REPORTS_CSV)
projections = pd.read_csv(PROJECTIONS_CSV)

print(f"Reports:      {len(reports)}")
print(f"Projections:  {len(projections)}")


#check columns
required_report_columns = {
    "uid",
    "findings",
    "impression",
}

required_projection_columns = {
    "uid",
    "filename",
    "projection",
}

missing_report_columns = (
    required_report_columns
    - set(reports.columns)
)

missing_projection_columns = (
    required_projection_columns
    - set(projections.columns)
)

if missing_report_columns:
    raise ValueError(
        "Missing columns in indiana_reports.csv: "
        f"{sorted(missing_report_columns)}"
    )

if missing_projection_columns:
    raise ValueError(
        "Missing columns in indiana_projections.csv: "
        f"{sorted(missing_projection_columns)}"
    )


# select frontals only
frontals = (
    projections[
        projections["projection"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("frontal")
    ]
    .sort_values(["uid", "filename"])
    .drop_duplicates(subset="uid")
    .copy()
)

print(f"Frontal studies: {len(frontals)}")


# combine reports with images
df = frontals.merge(
    reports,
    on="uid",
    how="inner",
)

print(f"After merge: {len(df)}")


# clean data
df["findings"] = (
    df["findings"]
    .fillna("")
    .astype(str)
)

df["impression"] = (
    df["impression"]
    .fillna("")
    .astype(str)
)

df = df[
    (df["findings"].str.strip() != "")
    |
    (df["impression"].str.strip() != "")
].copy()

def make_image_path(filename: Any) -> Path:
    return IMAGE_DIR / str(filename)


df["image_path"] = (
    df["filename"]
    .apply(make_image_path)
)

df = df[
    df["image_path"].apply(
        lambda path:
        isinstance(path, Path)
        and path.is_file()
    )
].copy()

print(f"Usable studies: {len(df)}")

df = df.head(N).copy()

print(
    f"Running grid search on "
    f"{len(df)} studies."
)

if len(df) == 0:
    raise RuntimeError(
        "No usable studies were found."
    )

print("\nLoading LLaVA-MED")

model_name = get_model_name_from_path(
    MODEL_NAME
)

if DEVICE.type == "cuda":
    model_dtype = torch.float16
else:
    model_dtype = torch.float32


tokenizer, model, image_processor, context_len = (
    load_pretrained_model(
        MODEL_NAME,
        None,
        model_name,
        load_8bit=False,
        load_4bit=True,
        device=str(DEVICE),
    )
)

model.eval()

print("LLaVA-MED loaded.")

original_prepare_inputs = (
    model.prepare_inputs_for_generation
)


def prepare_inputs_for_generation(
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    inputs_embeds_cd=None,
    images_cd=None,
    cd_alpha=None,
    cd_beta=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
):
    """
    Standard/original image.
    """

    return original_prepare_inputs(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
    )


def prepare_inputs_for_generation_cd(
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    inputs_embeds_cd=None,
    images_cd=None,
    cd_alpha=None,
    cd_beta=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
):
    """
    CD/noisy image.
    """

    if inputs_embeds_cd is not None:
        inputs_embeds = inputs_embeds_cd

    return original_prepare_inputs(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
    )


model.prepare_inputs_for_generation = (
    prepare_inputs_for_generation
)

model.prepare_inputs_for_generation_cd = (
    prepare_inputs_for_generation_cd
)


# prompt
PROMPT = (
    "Generate a radiology report for this chest X-ray. "
    "Include a Findings section and an Impression section. "
    "Describe only findings supported by the image. "
    "Do not invent clinical history or unsupported abnormalities."
)


def prepare_prompt():

    if "mistral_instruct" not in conv_templates:

        available_templates = (
            list(conv_templates.keys())
        )

        raise KeyError(
            "The conversation template "
            "'mistral_instruct' was not found.\n\n"
            f"Available templates:\n"
            f"{available_templates}"
        )

    conv = (
        conv_templates[
            "mistral_instruct"
        ].copy()
    )

    user_message = (
        "<image>\n"
        + PROMPT
    )

    conv.append_message(
        conv.roles[0],
        user_message,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    if not isinstance(
        input_ids,
        torch.Tensor,
    ):
        raise TypeError(
            "tokenizer_image_token() did not "
            "return a torch.Tensor."
        )

    return input_ids.unsqueeze(0).to(DEVICE)


# generate report
def generate_report(
    image_tensor,
    image_tensor_cd,
    image_size,
    input_ids,
    alpha,
    beta,
):
    """
    Generate one report for one image and one
    VCD parameter combination.
    """

    attention_mask = torch.ones_like(
        input_ids
    )

    (
        _,
        _,
        attn_mask,
        _,
        embeds,
        _,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        None,
        attention_mask,
        None,
        None,
        image_tensor.unsqueeze(0),
        image_sizes=[image_size],
    )

    if image_tensor_cd is None:

        embeds_cd = None

    else:

        (
            _,
            _,
            _,
            _,
            embeds_cd,
            _,
        ) = (
            model.prepare_inputs_labels_for_multimodal(
                input_ids,
                None,
                attention_mask,
                None,
                None,
                image_tensor_cd.unsqueeze(0),
                image_sizes=[image_size],
            )
        )

    with torch.inference_mode():

        output_ids = (
            MistralForCausalLM.generate(
                model,

                attention_mask=attn_mask,

                inputs_embeds=embeds,

                inputs_embeds_cd=embeds_cd,

                images_cd=image_tensor_cd,

                cd_alpha=alpha,

                cd_beta=beta,

                do_sample=True,

                temperature=0.7,

                top_p=0.9,

                max_new_tokens=512,

                use_cache=True,

                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )
        )

    report = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    return report

print("\nPreparing images...")

prepared_studies = []


for _, row in df.iterrows():

    uid = row["uid"]

    image_path = Path(
        row["image_path"]
    )

    print(
        f"Preparing {uid}: "
        f"{image_path.name}"
    )

    with Image.open(image_path) as img:

        image = img.convert("RGB")

    image_size = image.size

    # Original image processing
    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    if isinstance(
        image_tensor,
        list,
    ):
        image_tensor = image_tensor[0]

    if not isinstance(
        image_tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "process_images() did not return "
            "a torch.Tensor."
        )

    image_tensor = image_tensor.to(
        device=DEVICE,
        dtype=model_dtype,
    )

    input_ids = prepare_prompt()

    prepared_studies.append(
        {
            "uid": uid,
            "image_path": image_path,
            "image_size": image_size,
            "image_tensor": image_tensor,
            "input_ids": input_ids,
            "findings": row["findings"],
            "impression": row["impression"],
        }
    )


print(
    f"Prepared {len(prepared_studies)} studies."
)


# grid search
total_combinations = (
    len(ALPHA_VALUES)
    * len(BETA_VALUES)
    * len(NOISE_STEP_VALUES)
)

total_runs = (
    total_combinations
    * len(prepared_studies)
)

print("\n" + "=" * 80)
print("STARTING VCD GRID SEARCH")
print("=" * 80)

print(
    f"Alpha values:      {ALPHA_VALUES}"
)

print(
    f"Beta values:       {BETA_VALUES}"
)

print(
    f"Noise steps:       {NOISE_STEP_VALUES}"
)

print(
    f"Parameter combinations: "
    f"{total_combinations}"
)

print(
    f"Studies:            "
    f"{len(prepared_studies)}"
)

print(
    f"Total generations:  "
    f"{total_runs}"
)

print("=" * 80)


results = []

run_number = 0


# the grid 
for alpha in ALPHA_VALUES:

    for beta in BETA_VALUES:

        for noise_step in NOISE_STEP_VALUES:

            print("\n")
            print("#" * 80)

            print(
                f"ALPHA = {alpha}"
            )

            print(
                f"BETA = {beta}"
            )

            print(
                f"NOISE STEP = {noise_step}"
            )

            print("#" * 80)


            # process every study
            for study in prepared_studies:

                run_number += 1

                uid = study["uid"]

                print(
                    f"\nRun "
                    f"{run_number}/{total_runs}"
                )

                print(
                    f"UID: {uid}"
                )

                print(
                    f"alpha={alpha}, "
                    f"beta={beta}, "
                    f"noise={noise_step}"
                )
# Reset seed for every parameter combination.
               
                set_seed(SEED)

                if USE_CD:

                    image_tensor_cd = (
                        add_diffusion_noise(
                            study["image_tensor"],
                            noise_step,
                        )
                        .to(
                            device=DEVICE,
                            dtype=model_dtype,
                        )
                    )

                else:

                    image_tensor_cd = None

                try:

                    report = generate_report(
                        image_tensor=(
                            study["image_tensor"]
                        ),

                        image_tensor_cd=(
                            image_tensor_cd
                        ),

                        image_size=(
                            study["image_size"]
                        ),

                        input_ids=(
                            study["input_ids"]
                        ),

                        alpha=alpha,

                        beta=beta,
                    )


                    print(
                        "\nGENERATED REPORT:"
                    )

                    print(report)


                    error = ""


                except Exception as exc:

                    report = ""

                    error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    print(
                        "ERROR:",
                        error
                    )

                results.append(
                    {
                        "uid": uid,

                        "image_path": str(
                            study["image_path"]
                        ),

                        "alpha": alpha,

                        "beta": beta,

                        "noise_step": noise_step,

                        "seed": SEED,

                        "findings": (
                            study["findings"]
                        ),

                        "impression": (
                            study["impression"]
                        ),

                        "llavamed_report": report,

                        "error": error,
                    }
                )

                results_df = pd.DataFrame(
                    results
                )

                results_df.to_csv(
                    OUTPUT_CSV,
                    index=False,
                )

                del image_tensor_cd

                gc.collect()

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)


print("\n")
print("=" * 80)
print("GRID SEARCH FINISHED")
print("=" * 80)

print(
    f"Number of generated reports: "
    f"{len(results_df)}"
)

print(
    "Results saved to:"
)

print(OUTPUT_CSV)

print("=" * 80)
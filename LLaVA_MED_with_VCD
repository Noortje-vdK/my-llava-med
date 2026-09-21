"""
Author: GROUP 10 
Last changed: 2026/09/20

Aim script:

Python script is based on the test.py function in the GitHub setup by Noortje (based on LLaVA-MED)
Here changes are made such that it includes VCD

One note: it does not follow the example of the GithUb page of VCD. Namely, their implementation of VCD for LLaVA-MED is based on an old architecture of LLaVA-MED.
In the newer version (that we use) there are some changes in how the input is processed. The code below is based on the new input process.

In case there is a prefix with CD, it stands for the processing of the Contrastive Decoding image (so the image with noise applied)
"""

from pathlib import Path
from typing import Any
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import gc

# ------------------------------------------------- BELOW NEW

# Imports needed due to the VCD implementation
# MistralForCausalLM: a language model that is normally called within LLaVA-MED itself, but now we need to call it in this script
# set_seed: used for reproducibility of the sampled noise
from transformers import MistralForCausalLM, set_seed

# Import of functions by the VCD modules itself
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vcd_sample import evolve_vcd_sampling

# Settings needed for the VCD model
USE_CD: bool = True        # True, then CD ; False, then no CD
NOISE_STEP: int = 500      # Diffusion step of the distorted image (in VCD article between 0-999 was tested)
CD_ALPHA: float = 1.0      # Contrast strength,are
CD_BETA: float = 0.2       # Cutoff
SEED: int = 20             # For reprodubility of the random sampled noise

# Below replaces the transformer of the LLaVA-MED with the transformer of the VCD
# Difference between transformer LLaVA-MED and VCD: both do exactly the same, only VCD applies it parallel to the
# image with noise (CD image) and the normal image.
# This is adapted for the transformer 4.36.x.
if USE_CD:
    evolve_vcd_sampling()

# ------------------------------------------------- END NEW

gc.collect()
torch.cuda.empty_cache()

# Root folder downloaded from Kaggle
DATASET_ROOT: Path = Path(r"/kaggle/input/chest-xrays-indiana-university") # this is the path you may have to change to direct to the dataset
REPORTS_CSV: Path = DATASET_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV: Path = DATASET_ROOT / "indiana_projections.csv"
IMAGE_DIR: Path = DATASET_ROOT / "images" / "images_normalized"

# LLaVA-Med model
MODEL_NAME: str = "microsoft/llava-med-v1.5-mistral-7b"

# Number of studies to process, change this to desired nr of reports
N: int = 3

# Save output in the same folder as this Python script
OUTPUT_DIR = Path(__file__).resolve().parent

# Results including CD, then need a seperate output folder
if USE_CD:
    OUTPUT_CSV = OUTPUT_DIR / f"llavamed_results_{N}_vcd.csv"
else:
    OUTPUT_CSV = OUTPUT_DIR / f"llavamed_results_{N}.csv"


# device things
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print(
        "CUDA is not available. "
        "LLaVA-Med 7B will be very slow not fit")


# check data files

print("\nChecking data files")

required_files = [REPORTS_CSV,PROJECTIONS_CSV,IMAGE_DIR,]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path does not exist:\n{path}")

print("Dataset paths OK.")


# load dataset
print("\nLoading dataset")

reports: pd.DataFrame = pd.read_csv(REPORTS_CSV)
projections: pd.DataFrame = pd.read_csv(PROJECTIONS_CSV)

print(f"Reports:      {len(reports)}")
print(f"Projections:  {len(projections)}")


# check column sin data
required_report_columns = {
    "uid",
    "findings",
    "impression",}

required_projection_columns = {
    "uid",
    "filename",
    "projection",}

missing_report_columns = (
    required_report_columns - set(reports.columns))

missing_projection_columns = (
    required_projection_columns - set(projections.columns))

if missing_report_columns:
    raise ValueError(
        "Missing columns in indiana_reports.csv: "
        f"{sorted(missing_report_columns)}")

if missing_projection_columns:
    raise ValueError(
        "Missing columns in indiana_projections.csv: "
        f"{sorted(missing_projection_columns)}")


# select frontal xrays only
frontals: pd.DataFrame = (
    projections[
        projections["projection"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("frontal")
    ]
    .sort_values(["uid", "filename"])
    .drop_duplicates(subset="uid")
    .copy())

print(f"Frontal studies: {len(frontals)}")

# connect report and images
df: pd.DataFrame = frontals.merge(
    reports,
    on="uid",
    how="inner",)

print(f"After merge: {len(df)}")


# clean reference reports
df["findings"] = (
    df["findings"]
    .fillna("")
    .astype(str))

df["impression"] = (
    df["impression"]
    .fillna("")
    .astype(str))

df = df[
    (df["findings"].str.strip() != "")
    | (df["impression"].str.strip() != "")].copy()


def make_image_path(filename: Any) -> Path:
    """Create the full path to an X-ray image."""
    return IMAGE_DIR / str(filename)


df["image_path"] = df["filename"].apply(make_image_path)


# Keep only images that exist
df = df[
    df["image_path"].apply(
        lambda path: isinstance(path, Path) and path.is_file()
    )
].copy()

print(f"Usable studies: {len(df)}")


# select nr of studies 
df = df.head(N).copy()

print(f"Running LLaVA-Med on {len(df)} studies.")

if len(df) == 0:
    raise RuntimeError(
        "No usable studies were found. ")


# load model
print("\nLoading LLaVA-Med")

model_name: str = get_model_name_from_path(MODEL_NAME)

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
projector = model.model.mm_projector

# print("Projector class:", type(projector))

# for name, param in projector.named_parameters():
#     print(
#         name,
#         "shape=", tuple(param.shape),
#         "dtype=", param.dtype,
#         "device=", param.device,
#     )
# print("\nModel config:")
# print("mm_vision_tower:", model.config.mm_vision_tower)
# print("mm_hidden_size:", getattr(model.config, "mm_hidden_size", None))
# print("mm_projector_type:", getattr(model.config, "mm_projector_type", None))
# print("image_aspect_ratio:", getattr(model.config, "image_aspect_ratio", None))
# print("mm_vision_select_layer:", getattr(model.config, "mm_vision_select_layer", None))
# print("mm_vision_select_feature:", getattr(model.config, "mm_vision_select_feature", None))

# print("LLaVA-Med loaded.")
# print("Model device:", next(model.parameters()).device)
# print("Context length:", context_len)

vision_tower = model.get_vision_tower()

# print("\n===== VISION TOWER DEBUG =====")
# print("Vision tower type:", type(vision_tower))
# print("Vision tower loaded:", vision_tower.is_loaded)
# print("Vision tower device:", next(vision_tower.parameters()).device)
# print("Vision tower dtype:", next(vision_tower.parameters()).dtype)

# print("\nProjector:")
# print("Projector device:", next(model.model.mm_projector.parameters()).device)
# print("Projector dtype:", next(model.model.mm_projector.parameters()).dtype)

# print("\nModel:")
# print("Model type:", type(model))
# print("Model device:", next(model.parameters()).device)


# ------------------------------------------------- BELOW NEW
# model.prepare_inputs_for_generation is a part of the LLaVA model.
# Here, it should be applied twice, namely to noraml and CD image. For clearity, label below is used.

original_prepare_inputs = model.prepare_inputs_for_generation

# Function is for processing on the image without CD
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
    output_hidden_states=None):
    """
    This function calls the model.prepare_inputs_for_generation function inside the model class
    as provided in the LLaVA-MED algorithm.
    This function serves as selecting the correct input for that function.
    This function processes the image without noise / CD.
    """
    return original_prepare_inputs(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache)

# Function is for processing on the image with CD
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
    output_hidden_states=None):
    """
    This function calls the model.prepare_inputs_for_generation function inside the model class
    as provided in the LLaVA-MED algorithm.
    This function serves as selecting the correct input for that function.
    This function processes the image with noise / CD.
    """
    if inputs_embeds_cd is not None:
        inputs_embeds = inputs_embeds_cd
    return original_prepare_inputs(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache)

# model is a class inside the LLaVA-MED
# Here, the prepared input values are added as a variable for the class for 
# both the CD and non-CD image.
model.prepare_inputs_for_generation = prepare_inputs_for_generation
model.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd

# ------------------------------------------------- END NEW

# generate rpeort
def generate_report(image_path: Path) -> str:
    """
    Generate a radiology report from one chest X-ray.
    """

    print(f"\nProcessing: {image_path}")

    # VCD: seed per study, not once per run, so study i starts from the same
    # random state in every run -- otherwise a difference between a VCD run and
    # a baseline run could just as well be sampling noise. Same as VCD's eval
    # script, which calls set_seed(args.seed) too.
    set_seed(SEED)

    # load image and check it
    with Image.open(image_path) as img:
        image: Image.Image = img.convert("RGB")

    print("Image:",image_path.name,
    "size:",image.size,
    "mode:",image.mode,
    "mean pixel:",sum(image.convert("L").getdata()) /(image.width * image.height),)

    # process image
    image_tensor = process_images(
        [image],
        image_processor,
        model.config,)

    if isinstance(image_tensor, list):
        image_tensor = image_tensor[0]

    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(
            "process_images() did not return a torch.Tensor."
        )

    # ------------------------------------------------- BELOW NEW
    # This piece is based on the code as provided by the VCD as example
    # It serves as adding noise to the image.
    if USE_CD:
        image_tensor_cd = add_diffusion_noise(image_tensor, NOISE_STEP).to(
            device=DEVICE,
            dtype=model_dtype,
        )
    else:
        image_tensor_cd = None
    # ------------------------------------------------- END NEW
    
    image_tensor = image_tensor.to(
        device=DEVICE,
        dtype=model_dtype,
    )

    # prompt
    prompt: str = (
        "Generate a radiology report for this chest X-ray. "
        "Include a Findings section and an Impression section. "
        "Describe only findings supported by the image. "
        "Do not invent clinical history or unsupported abnormalities.")

    # intialize conversation structure, it is a little different than in medgemma
    if "mistral_instruct" not in conv_templates:
        available_templates = list(conv_templates.keys())

        raise KeyError(
            "The conversation template 'mistral_instruct' "
            "was not found.\n\n"
            f"Available templates:\n{available_templates}"
        )

    conv = conv_templates["mistral_instruct"].copy()

    user_message = "<image>\n" + prompt

    conv.append_message(
        conv.roles[0],
        user_message,)

    conv.append_message(
        conv.roles[1],
        None,)

    prompt_text: str = conv.get_prompt()

    # tokenize
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",)

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            "tokenizer_image_token() did not return a torch.Tensor.")

    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    # generate report
    attention_mask = torch.ones_like(input_ids)

    # ------------------------------------------------- BELOW NEW
    # Code below is based on the code provided by LLaVA-MED
    # Code is set here, since now two input images should be run parallel
    # LLaVA-MED uses model.generate(). This function is here written out
    
    # First performing it for the image without CD, preparing correct input
    _, _, attn_mask, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, attention_mask, None, None,
        image_tensor.unsqueeze(0), image_sizes=[image.size])

    # Then performing for the image with CD, preparing correct input
    if image_tensor_cd is None:
        embeds_cd = None
    else:
        _, _, _, _, embeds_cd, _ = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, attention_mask, None, None,
            image_tensor_cd.unsqueeze(0), image_sizes=[image.size])

    # Running both images through the model, with the corresponding CD values
    # Some parameters here (e.g. temperature, max_new_tokens) can be replaced to above
    # such that an userinterface can be made in the end :-)
    with torch.inference_mode():

        output_ids = MistralForCausalLM.generate(
            model,
            attention_mask=attn_mask,
            inputs_embeds=embeds,
            inputs_embeds_cd=embeds_cd, # If no VCD, then None
            images_cd=image_tensor_cd, # If no VCD, then None
            cd_alpha=CD_ALPHA,
            cd_beta=CD_BETA,
            do_sample=True, # For VCD, it should stay True
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512, # Length of reports
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # ------------------------------------------------- END NEW

    print("Output token IDs:")
    print(output_ids[0].tolist())
    
    print("\nDEBUG:")
    print("input_ids shape:", input_ids.shape)
    print("output_ids shape:", output_ids.shape)

    report = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    print("\nGENERATED REPORT:")
    print(repr(report))

    return report

# run inference 
generated_reports: list[str] = []

for _, row in df.iterrows():

    uid = row["uid"]
    image_path = Path(row["image_path"])

    try:

        report = generate_report(image_path)

        generated_reports.append(report)

        print("\n" + "=" * 80)
        print(f"Study UID: {uid}")
        print("\nLLaVA-Med report:")
        print(report)
        print("=" * 80)

    except Exception as exc:

        error_message = (
            f"ERROR processing {uid}: "
            f"{type(exc).__name__}: {exc}"
        )

        print(error_message)

        generated_reports.append(error_message)


# save results
df["llavamed_report"] = generated_reports

print("\nColumns being saved:")
print(df.columns.tolist())

print("\nGenerated reports:")
for i, report in enumerate(generated_reports):
    print(f"\n--- Report {i + 1} ---")
    print(report)

print("\nDataFrame preview:")
print(df[["uid", "llavamed_report"]].to_string())

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\nFinished.")
print("Results saved to:")
print(OUTPUT_CSV)

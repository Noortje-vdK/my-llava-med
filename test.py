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

# generate rpeort
def generate_report(image_path: Path) -> str:
    """
    Generate a radiology report from one chest X-ray.
    """

    print(f"\nProcessing: {image_path}")

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
    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=image_tensor.unsqueeze(0),
            image_sizes=[image.size],
            do_sample=True, # set to false if you dont want to use temp and top-p
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512, # change this to allow longer/shorter reports
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
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

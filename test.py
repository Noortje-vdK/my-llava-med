
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
    get_model_name_from_path,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

import gc

gc.collect()
torch.cuda.empty_cache()
# ============================================================
# 1. CONFIGURATION
# ============================================================

# Root folder downloaded from Kaggle
DATASET_ROOT: Path = Path(
    r"C:\Users\20234103\.cache\kagglehub\datasets\raddar\chest-xrays-indiana-university\versions\2"
)

REPORTS_CSV: Path = DATASET_ROOT / "indiana_reports.csv"
PROJECTIONS_CSV: Path = DATASET_ROOT / "indiana_projections.csv"
IMAGE_DIR: Path = DATASET_ROOT / "images" / "images_normalized"

# LLaVA-Med model
MODEL_NAME: str = "microsoft/llava-med-v1.5-mistral-7b"

# Number of studies to process
N: int = 1

# Output file
OUTPUT_CSV: Path = DATASET_ROOT / f"llavamed_results_{N}.csv"


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print(
        "WARNING: CUDA is not available. "
        "LLaVA-Med 7B will be very slow and may require too much RAM."
    )


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\nChecking dataset files...")

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


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

reports: pd.DataFrame = pd.read_csv(REPORTS_CSV)
projections: pd.DataFrame = pd.read_csv(PROJECTIONS_CSV)

print(f"Reports:      {len(reports)}")
print(f"Projections:  {len(projections)}")


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

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
    required_report_columns - set(reports.columns)
)

missing_projection_columns = (
    required_projection_columns - set(projections.columns)
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


# ============================================================
# 6. KEEP FRONTAL CHEST X-RAYS
# ============================================================

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
    .copy()
)

print(f"Frontal studies: {len(frontals)}")


# ============================================================
# 7. MERGE REPORTS + IMAGES
# ============================================================

df: pd.DataFrame = frontals.merge(
    reports,
    on="uid",
    how="inner",
)

print(f"After merge: {len(df)}")


# ============================================================
# 8. CLEAN REFERENCE REPORTS
# ============================================================

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
    | (df["impression"].str.strip() != "")
].copy()


# ============================================================
# 9. CREATE IMAGE PATH
# ============================================================

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


# ============================================================
# 10. SELECT N STUDIES
# ============================================================

df = df.head(N).copy()

print(f"Running LLaVA-Med on {len(df)} studies.")


if len(df) == 0:
    raise RuntimeError(
        "No usable studies were found. "
        "Check IMAGE_DIR, the CSV files, and the image filenames."
    )


# ============================================================
# 11. LOAD LLAVA-MED
# ============================================================

print("\nLoading LLaVA-Med...")

model_name: str = get_model_name_from_path(MODEL_NAME)

# Use CUDA when available.
# float16 is appropriate for most NVIDIA GPUs.
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

# Move model if necessary.

model.eval()

print("LLaVA-Med loaded.")
print("Model device:", next(model.parameters()).device)
print("Context length:", context_len)


# ============================================================
# 12. GENERATE ONE RADIOLOGY REPORT
# ============================================================

def generate_report(image_path: Path) -> str:
    """
    Generate a radiology report from one chest X-ray.
    """

    print(f"\nProcessing: {image_path}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    with Image.open(image_path) as img:
        image: Image.Image = img.convert("RGB")

    # --------------------------------------------------------
    # Process image
    # --------------------------------------------------------

    image_tensor = process_images(
        [image],
        image_processor,
        model.config,
    )

    # Different LLaVA versions can return either a list
    # or a tensor.
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

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt: str = (
        "Generate a radiology report for this chest X-ray. "
        "Include a Findings section and an Impression section. "
        "Describe only findings supported by the image. "
        "Do not invent clinical history or unsupported abnormalities."
    )

    # --------------------------------------------------------
    # Conversation
    # --------------------------------------------------------

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
        user_message,
    )

    conv.append_message(
        conv.roles[1],
        None,
    )

    prompt_text: str = conv.get_prompt()

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            "tokenizer_image_token() did not return a torch.Tensor."
        )

    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
        )

    # --------------------------------------------------------
    # Decode only newly generated tokens
    # --------------------------------------------------------

    generated_ids = output_ids[
        :, input_ids.shape[1]:
    ]

    report: str = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return report


# ============================================================
# 13. RUN INFERENCE
# ============================================================

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


# ============================================================
# 14. SAVE RESULTS
# ============================================================

df["llavamed_report"] = generated_reports

df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\nFinished.")
print("Results saved to:")
print(OUTPUT_CSV)

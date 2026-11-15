import argparse
import json
import math
from pathlib import Path

import open_clip
import torch
from PIL import Image, ImageOps
from llama_index.core.schema import Document


MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIMENSION = 512
PREPROCESSING = "white_background_square_pad_v1"


def prepare_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as original:
        rgba = original.convert("RGBA")

        background = Image.new(
            "RGBA",
            rgba.size,
            (255, 255, 255, 255),
        )

        rgb = Image.alpha_composite(
            background,
            rgba,
        ).convert("RGB")

    side = max(rgb.size)

    return ImageOps.pad(
        rgb,
        (side, side),
        color="white",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for visual documents."
    )
    parser.add_argument(
        "visual_documents_json",
        type=Path,
        help="Path to the LlamaIndex visual documents JSON.",
    )

    args = parser.parse_args()
    input_path = args.visual_documents_json.resolve()

    if not input_path.is_file():
        parser.error(f"JSON file not found: {input_path}")

    records = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    if not isinstance(records, list) or not records:
        parser.error("Expected a non-empty list of documents.")

    documents = [
        Document.from_dict(record)
        for record in records
    ]

    if len({document.id_ for document in documents}) != len(documents):
        parser.error("The input contains duplicate document IDs.")

    for document in documents:
        resource = document.image_resource

        if resource is None or resource.path is None:
            parser.error("A document has no local image resource.")

        if not resource.path.is_file():
            parser.error(f"Image not found: {resource.path}")

    print(
        "Loading CLIP. The first run downloads the model weights.",
        flush=True,
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        device="cpu",
        precision="fp32",
    )
    model.eval()

    embedded_records = []

    with torch.inference_mode():
        for position, document in enumerate(documents, start=1):
            image_path = document.image_resource.path

            with prepare_image(image_path) as image:
                image_tensor = preprocess(image).unsqueeze(0)

            features = model.encode_image(
                image_tensor,
                normalize=True,
            )

            if tuple(features.shape) != (1, EMBEDDING_DIMENSION):
                parser.error(
                    f"Visual {position}: unexpected embedding shape."
                )

            if not torch.isfinite(features).all().item():
                parser.error(
                    f"Visual {position}: invalid embedding values."
                )

            vector = features[0].tolist()
            squared_norm = sum(value * value for value in vector)

            if not math.isclose(
                squared_norm,
                1.0,
                abs_tol=1e-4,
            ):
                parser.error(
                    f"Visual {position}: embedding is not normalized."
                )

            embedded_records.append(
                {
                    "visual_id": document.id_,
                    "image_path": document.metadata["image_path"],
                    "metadata": document.metadata,
                    "embedding_model": (
                        f"open_clip:{MODEL_NAME}:{PRETRAINED}"
                    ),
                    "preprocessing": PREPROCESSING,
                    "embedding": vector,
                }
            )

            print(
                f"Embedded visual {position}/{len(documents)} "
                f"- page {document.metadata['page']}",
                flush=True,
            )

    output_path = input_path.with_name(
        f"{input_path.stem}.clip_embeddings.json"
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            embedded_records,
            output_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"\nVisuals embedded: {len(embedded_records)}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
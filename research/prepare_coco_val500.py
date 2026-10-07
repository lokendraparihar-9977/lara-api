from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMAGES = (
    PROJECT_ROOT
    / "datasets"
    / "coco-val2017-mini500"
    / "images"
    / "val2017"
)

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "coco-val2017-mini500"
    / "annotations"
    / "instances_val2017_mini500.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "research"
    / "evaluation"
    / "coco_val500"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"

DATASET_YAML = OUTPUT_ROOT / "data.yaml"


def convert_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    Convert COCO bbox:
        [x_min, y_min, width, height]

    to normalized YOLO bbox:
        [x_center, y_center, width, height]
    """

    x_min, y_min, width, height = bbox

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return (
        x_center / image_width,
        y_center / image_height,
        width / image_width,
        height / image_height,
    )


def main() -> None:
    print("=" * 70)
    print("PREPARING COCO VAL2017 MINI-500")
    print("=" * 70)

    if not SOURCE_IMAGES.exists():
        raise FileNotFoundError(
            f"Image directory not found: {SOURCE_IMAGES}"
        )

    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    with ANNOTATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"COCO images:       {len(images)}")
    print(f"COCO annotations:  {len(annotations)}")
    print(f"COCO categories:   {len(categories)}")

    # ---------------------------------------------------------
    # COCO category ID -> contiguous YOLO class ID
    # ---------------------------------------------------------

    categories_sorted = sorted(
        categories,
        key=lambda category: category["id"],
    )

    category_to_yolo = {
        category["id"]: index
        for index, category in enumerate(categories_sorted)
    }

    class_names = [
        category["name"]
        for category in categories_sorted
    ]

    # ---------------------------------------------------------
    # Create output directories
    # ---------------------------------------------------------

    OUTPUT_IMAGES.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_LABELS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Group annotations by image ID
    # ---------------------------------------------------------

    annotations_by_image: dict[int, list[dict]] = {}

    for annotation in annotations:
        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    copied_images = 0
    written_annotations = 0

    # ---------------------------------------------------------
    # Convert dataset
    # ---------------------------------------------------------

    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_width = image_info["width"]
        image_height = image_info["height"]

        source_image = SOURCE_IMAGES / file_name
        destination_image = OUTPUT_IMAGES / file_name

        if not source_image.exists():
            raise FileNotFoundError(
                f"Image referenced by annotations is missing: "
                f"{source_image}"
            )

        shutil.copy2(
            source_image,
            destination_image,
        )

        copied_images += 1

        label_file = (
            OUTPUT_LABELS
            / Path(file_name).with_suffix(".txt").name
        )

        lines: list[str] = []

        for annotation in annotations_by_image.get(
            image_id,
            [],
        ):
            # Ignore COCO crowd annotations for this
            # YOLO-format evaluation dataset.
            if annotation.get("iscrowd", 0):
                continue

            bbox = annotation["bbox"]

            (
                x_center,
                y_center,
                bbox_width,
                bbox_height,
            ) = convert_bbox(
                bbox,
                image_width,
                image_height,
            )

            category_id = annotation["category_id"]

            class_id = category_to_yolo[category_id]

            lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bbox_width:.6f} "
                f"{bbox_height:.6f}"
            )

            written_annotations += 1

        label_file.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    # ---------------------------------------------------------
    # Write Ultralytics dataset configuration
    # ---------------------------------------------------------

    yaml_lines = [
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: images",
        "val: images",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(class_names):
        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    DATASET_YAML.write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"Images copied:       {copied_images}")
    print(f"Annotations written: {written_annotations}")
    print(f"Classes:             {len(class_names)}")
    print()
    print(f"Images:              {OUTPUT_IMAGES}")
    print(f"Labels:              {OUTPUT_LABELS}")
    print(f"Dataset YAML:        {DATASET_YAML}")
    print("=" * 70)


if __name__ == "__main__":
    main()
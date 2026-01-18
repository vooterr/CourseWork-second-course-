import csv

from pathlib import Path
from PIL import Image

def parse_species(line: str) -> str:
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError("Invalid annotation line")

    species_id = parts[2]

    if species_id == "1":
        return "cat"
    elif species_id == "2":
        return "dog"
    else:
        raise ValueError(f"Unknown species id: {species_id}")
    

def extract_breed_name(image_id: str) -> str:
    """
    Abyssinian_100 → abyssinian
    american_bulldog_12 → american bulldog
    """
    name = image_id.rsplit("_", 1)[0]
    return name.replace("_", " ").lower()


def parse_annotations(annotation_file: str):
    samples = []
    classes = set()

    with open(annotation_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            image_id, class_id, species_id, breed_id = line.strip().split()

            breed = extract_breed_name(image_id)
            samples.append(image_id)
            classes.add(breed)

    return samples, sorted(classes)


def build_breed_classification_dataset(
    images_dir: str,
    annotation_file: str
):
    image_ids, classes = parse_annotations(annotation_file)
    dataset = []

    for image_id in image_ids:
        image_path = Path(images_dir) / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        dataset.append({
            "image_path": image_id,
            "image": image,
            "prompt": (
                "You are given an image of an animal.\n"
                "Your task is to identify its breed.\n"
                "Answer with a SINGLE breed name."
            ),
            "label": extract_breed_name(image_id),
            "classes": classes
        })

    return dataset


if __name__ == "__main__":
    dataset = build_breed_classification_dataset(
        images_dir="./data/classification/oxford_pet/images",
        annotation_file="./data/classification/oxford_pet/annotations/list.txt"
    )

    print("Rows =", len(dataset))          # ~7390
    print("Amount classes =", len(dataset[0]["classes"]))  # 37

    csv_dataset = {}
    dirs = "data/classification/oxford_pet/annotations"
    with open(dirs + "/annotations.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for row in dataset:
            writer.writerow([row["image_path"], "_".join(row["label"].split())])

    with open(dirs + "/classes.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for row in dataset[0]["classes"]:
            writer.writerow(["_".join(row.split())])

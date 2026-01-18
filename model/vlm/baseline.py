from model.vlm.model import LlavaVLM
from model.vlm.abstract_model import ModelConfig

from PIL import Image

import csv
import pathlib

def verification(image_path: str, max_new_tokens: int = 128) -> str:
    config = ModelConfig(device="cuda", dtype="float16")
    vlm = LlavaVLM(config=config)
    image = Image.open(image_path)
    
    vlm.load()
    inputs = vlm.preprocess(image=image, prompt="Identify the breed of the animal in the photo")
    outputs = vlm.generate(inputs=inputs, max_new_tokens=max_new_tokens)

    print(outputs)


def base_verification(images_dir: str, max_new_tokens: int = 128, max_img: int = 5):
    classes = []
    with open("data/classification/oxford_pet/annotations/classes.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            classes.append(row[0])
            
    classes = [" ".join(cls.split("_")) for cls in classes]
    
    prompt = (
        "Identify the dog breed shown in the image."
        "Output format: ONE WORD ONLY."
        f"Answer using ONLY one of the following classes: "
        f"{', '.join(classes)}."
        
    )
    
    
    path = pathlib.Path(images_dir)
    count = 0
    
    config = ModelConfig(device="cuda", dtype="float16")
    vlm =  LlavaVLM(config=config);
    vlm.load()
    
    for img in path.iterdir():
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        with Image.open(img) as image:
            inputs = vlm.preprocess(image=image, prompt=prompt)
            outputs = vlm.generate(inputs=inputs, max_new_tokens=max_new_tokens)

            print(outputs, img.name)
        count+=1
        if count == max_img:
            break


if __name__ == "__main__":
    base_verification(images_dir="data/classification/oxford_pet/images")
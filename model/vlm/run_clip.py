from model.vlm.model import CLIPVLM
from model.vlm.abstract_model import ModelConfig

from PIL import Image

import csv
import pathlib

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
    vlm = CLIPVLM(config=config)
    vlm.load()
    
    prompt = vlm.generate_prompt(classes)
    
    for img in path.iterdir():
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        with Image.open(img) as image:
            inputs = vlm.preprocess(image=image, text_prompts=prompt)
            outputs, score = vlm.generate(inputs=inputs)

            print(classes[outputs], img.name)
            print("Score:", score)
        count+=1
        if count == max_img:
            break

if __name__ == "__main__":
    base_verification(images_dir="data/classification/oxford_pet/images")
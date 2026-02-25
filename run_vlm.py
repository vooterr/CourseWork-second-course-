from model.vlm.model import Florence2VLM, QwenVLM, DinoVLM
from model.vlm.abstract_model import ModelConfig
from metrics.detection import evaluate_dataset
from utils.visualizer import draw_and_save

from PIL import Image

config = ModelConfig(device="cuda", dtype="float16", quantization="4bit")
model = DinoVLM(config=config)

model.load()
img = Image.open("./data/classification/oxford_pet/pug_2.jpg")

output = model.detect(image=img, prompt="Pug.")

draw_and_save(image=img, ground_truths=output, predictions=output, filename="Pug.jpg")

print(output)

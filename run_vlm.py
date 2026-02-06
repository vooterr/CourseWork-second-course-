from model.vlm.model import Florence2VLM, QwenVLM
from model.vlm.abstract_model import ModelConfig
from metrics.detection import evaluate_dataset

from PIL import Image

config = ModelConfig(device="cuda", dtype="float16", quantization="4bit")
model = QwenVLM(config=config)

model.load()
img = Image.open("./data/classification/oxford_pet/pug_1.jpg")
print(img)
output = model.detect(image=img, prompt="<OD>")

print(output)

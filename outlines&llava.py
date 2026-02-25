import torch 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

import requests
from PIL import Image



def detect() -> str:
    device = Accelerator().device

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    text_prompt = "a cat. a remote control"

    inputs = processor(
        images=image, 
        text=text_prompt, 
        return_tensors="pt"

    ).to(device)


    with torch.no_grad():
        outputs = model(**inputs)


    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]
    for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
        box = [round(x, 2) for x in box.tolist()]
        print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
    return results


if __name__ == "__main__":
    output = detect()

    print(output)

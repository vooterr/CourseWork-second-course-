import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForObjectDetection

# 1. Загружаем модель и процессор
# Мы используем "tiny" версию для скорости, есть и более крупные
model_id = "microsoft/glip" 
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)

# 2. Готовим картинку и текстовый запрос
image = Image.open("street_photo.jpg").convert("RGB")

# В тексте мы перечисляем всё, что хотим найти, через точку.
# GLIP очень чувствителен к формулировкам.
text_prompt = "detect the silver laptop. find a person wearing a red jacket. a cracked phone screen."

# 3. Превращаем данные в тензоры
inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)

# 4. Прогон через нейросеть (Inference)
with torch.no_grad():
    outputs = model(**inputs)

# 5. Пост-обработка результатов
# Модель выдает кучу боксов, нам нужно отфильтровать их по порогу уверенности (threshold)
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.5, # Вероятность выше 50%
    target_sizes=target_sizes
)[0]

# 6. Вывод результата
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Нашел объект: '{label}' с уверенностью {round(score.item(), 3)} по координатам {box}")
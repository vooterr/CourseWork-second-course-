import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from outlines.models.transformers import Transformers
from qwen_vl_utils import process_vision_info
from PIL import Image
from pydantic import BaseModel
from typing import List

# --- 1. Адаптер ---
class QwenVLAdapter(Transformers):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self.vision_inputs = {}

    def set_vision_inputs(self, inputs):
        self.vision_inputs = {k: v for k, v in inputs.items() if v is not None}

    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        # Важный момент: Qwen2.5 может капризничать с attention_mask при ручном расширении
        # Но обычно добавление pixel_values достаточно
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            **kwargs,
            **self.vision_inputs 
        }
        return self.model(**forward_kwargs)

# --- 2. Загрузка ---
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

hf_model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="cuda",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = QwenVLAdapter(hf_model, processor.tokenizer)

# --- 3. Данные ---
try:
    image = Image.open('test.jpg')
except:
    print("Используем заглушку!")
    image = Image.new('RGB', (600, 400), color='red')

# Упростим промпт для теста
prompt_text = (
    "Detect the cat. Return JSON with bbox [ymin, xmin, ymax, xmax]."
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }
]

# --- 4. МАГИЯ ИСПРАВЛЕНИЯ (The Fix) ---

# Шаг А: Создаем черновой шаблон
text_raw = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Шаг Б: Процессор считает патчи и создает правильные input_ids
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text_raw],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# Шаг В: Декодируем расширенные токены обратно в строку!
# Теперь строка будет содержать много-много спецтокенов картинки вместо одного
# skip_special_tokens=False ОБЯЗАТЕЛЬНО, иначе он удалит токены картинки
prompt_expanded = processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=False)

# Проверка (для отладки)
# print(f"Длина исходного текста: {len(text_raw)}")
# print(f"Длина расширенного текста: {len(prompt_expanded)}") 

# Загружаем тензоры
model.set_vision_inputs({
    "pixel_values": inputs.pixel_values,
    "image_grid_thw": inputs.image_grid_thw,
})

# --- 5. Генерация ---
class Detection(BaseModel):
    label: str
    bbox: List[int]

class Response(BaseModel):
    detections: List[Detection]

print("Запуск генерации с правильным количеством токенов...")

# Передаем prompt_expanded, а не text_raw
result = model(
    prompt_expanded, 
    Response, 
    max_new_tokens=1024
)

print(f"Результат: {result}")

result = Detection.model_validate_json(result)
# Денормализация (если bbox нашлись)
if result.detections:
    width, height = image.size
    for det in result.detections:
        ymin, xmin, ymax, xmax = det.bbox
        # Защита от выхода за границы 1000
        ymin, xmin, ymax, xmax = [min(max(x, 0), 1000) for x in [ymin, xmin, ymax, xmax]]
        
        abs_box = [
            int(xmin * width / 1000),
            int(ymin * height / 1000),
            int(xmax * width / 1000),
            int(ymax * height / 1000)
        ]
        print(f"Объект: {det.label}, Координаты: {abs_box}")

import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq,
    AutoModelForZeroShotObjectDetection, Qwen2VLForConditionalGeneration
)
from PIL import Image

from typing import List, Optional

from model.vlm.abstract_model import BaseVLM, Detection


# --- Florence-2 ---
class Florence2VLM(BaseVLM):
    model_id = "microsoft/Florence-2-large"
    device = None
    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = torch.cuda if self.config.device == "auto" and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.config.dtype == "float16" and self.device == "cuda" else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            attn_implementation="eager"
        ).to(self.device).eval()

    def detect(self, image: Image.Image, classes: Optional[List[str]] = None) -> List[Detection]:
        # 1. Выбор задачи (Task)
        if classes:
            task = "<CAPTION_TO_PHRASE_GROUNDING>"
            text_input = task + " " + ". ".join(classes)
        else:
            task = "<OD>"
            text_input = task

        # 2. Подготовка и генерация
        inputs = self.processor(text=text_input, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Приведение типов для float16
        if self.config.dtype == "float16" and self.device == "cuda":
             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False 
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # 3. Парсинг
        try:
            parsed = self.processor.post_process_generation(
                generated_text, task=task, image_size=image.size
            )
            data = parsed.get(task, {})
            
            detections = []
            for label, bbox in zip(data.get("labels", []), data.get("bboxes", [])):
                detections.append(Detection(label=label, bbox=bbox, score=1.0))
            return detections
        except Exception as e:
            print(f"Florence parsing error: {e}")
            return []


# --- Grounding DINO ---
class DinoVLM(BaseVLM):
    model_id = "IDEA-Research/grounding-dino-tiny"

def load(self) -> None:
    self.device = "cuda" if (self.config.device in ['cuda', 'auto'] and torch.cuda.is_available()) else "cpu"
    self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
    dtype = torch.float16 if self.config.dtype == "float16" else torch.float32

    self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
        self.model_id, 
        device_map=self.device, 
        torch_dtype=dtype, 
        cache_dir=self.cache_dir
    ).eval()

    def detect(self, image: Image.Image, classes: Optional[List[str]] = None) -> List[Detection]:
        if not classes:
            print("DINO requires specific classes.")
            return []
            
        # DINO требует формат "class . class ."
        prompt = " . ".join(classes) + "."
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Пост-процессинг встроен в библиотеку transformers для DINO
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )[0]

        detections = []
        for label, box, score in zip(results['text_labels'], results['boxes'], results['scores']):
            detections.append(Detection(
                label=label,
                bbox=box.round().int().tolist(),
                score=score.item()
            ))
        return detections


# --- Kosmos-2 ---
class KosmosVLM(BaseVLM):
    model_id = "microsoft/kosmos-2-patch14-224"

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device, 
            cache_dir=self.cache_dir
        ).to(self.device).eval()

    def detect(self, image: Image.Image, classes: Optional[List[str]] = None) -> List[Detection]:
        prompt_text = "Describe this image"
        if classes:
            # Kosmos понимает запросы на поиск в формате grounding
            prompt_text = "Find " + ", ".join(classes)
            
        final_prompt = f"<grounding> {prompt_text}"
        
        inputs = self.processor(text=final_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        if "pixel_values" in inputs and self.device == "cuda":
             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        _, entities = self.processor.post_process_generation(generated_text)

        detections = []
        for label, _, boxes in entities:
            for box in boxes:
                detections.append(Detection(label=label, bbox=list(box), score=1.0))
        return detections


# --- Qwen2-VL ---
class QwenVLM(BaseVLM):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
            device_map=self.device,
            cache_dir=self.cache_dir
        ).eval()

    def detect(self, image: Image.Image, classes: Optional[List[str]] = None) -> List[Detection]:
        prompt_text = "Detect objects."
        if classes:
            prompt_text = f"Detect {', '.join(classes)}"

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]
        }]
        
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=[text_input], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # Обрезка промпта из ответа
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # TODO: Реализовать парсинг координат для Qwen. 
        # Qwen обычно возвращает координаты в формате (0-1000) внутри текста.
        # Для полноценной детекции требуется ручной regex-парсинг вывода, 
        # так как стандартного post_process_object_detection у него может не быть в этой версии.
        
        return []

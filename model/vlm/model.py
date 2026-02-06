from model.vlm.abstract_model import BaseVLM, ModelConfig, Detection

import torch
import re
from typing import List, Any, Dict, Union
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    LlavaForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)




class LlavaVLM(BaseVLM):
    model_id = "llava-hf/llava-1.5-7b-hf"

    def _get_torch_dtype(self):
        mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        return mapping.get(self.config.dtype, torch.float16)

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            cache_dir=self.cache_dir
        )
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._get_torch_dtype(),
            cache_dir=self.cache_dir
        ).to(self.device)
        self.model.eval()
        print(f"{self.model_id} loaded.")

    def preprocess(self, image: Image.Image, prompt: str) -> Any:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate(self, inputs: Any, max_new_tokens: int = 200, **kwargs) -> str:
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        return self.processor.decode(outputs[0][2:], skip_special_tokens=True)

    def parse_coordinates(self, raw_output: str) -> List[Detection]:

        return []


class CLIPVLM(BaseVLM):
    model_id = "openai/clip-vit-large-patch14"

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image, prompt: str) -> Any:

        classes = [c.strip() for c in prompt.split(",")]
        inputs = self.processor(
            text=classes,
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs["classes_list"] = classes 
        return inputs.to(self.device)

    def generate(self, inputs: Any, **kwargs) -> str:
        classes = inputs.pop("classes_list")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            pred_idx = probs.argmax().item()
            
        return classes[pred_idx] 

    def parse_coordinates(self, raw_output: Any) -> List[Detection]:

        return []


class Florence2VLM(BaseVLM):
    model_id = "microsoft/Florence-2-large"

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.config.dtype == "float16" and self.device == "cuda" else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=dtype, 
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            attn_implementation="eager"
        ).to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image, prompt: str) -> Any:

        self._current_image_size = image.size 
        

        final_prompt = prompt if prompt else "<OD>"
        
        inputs = self.processor(text=final_prompt, images=image, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.config.dtype == "float16" and self.device == "cuda":
             inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
             
        return inputs

    def generate(self, inputs: Any, **kwargs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                use_cache=False
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    def parse_coordinates(self, raw_output: str) -> List[Detection]:

        task = "<OD>" 
        if "<CAPTION_TO_PHRASE_GROUNDING>" in raw_output: task = "<CAPTION_TO_PHRASE_GROUNDING>"
        
        try:
            parsed = self.processor.post_process_generation(
                raw_output, 
                task=task, 
                image_size=self._current_image_size
            )
            

            key = task
            detections = []
            if key in parsed:
                data = parsed[key]
                labels = data.get("labels", [])
                bboxes = data.get("bboxes", [])
                for label, bbox in zip(labels, bboxes):
                    detections.append(Detection(label=label, bbox=bbox, score=1.0))
            return detections
        except Exception as e:
            print(f"Error parsing Florence output: {e}")
            return []


class QwenVLM(BaseVLM):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.model.eval()

    def preprocess(self, image: Image.Image, prompt: str) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            images=[image],
            text=[text_input],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        return inputs

    def generate(self, inputs: Any, max_new_tokens: int = 128, **kwargs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def parse_coordinates(self, raw_output: str) -> List[Detection]:

        return []


class KosmosVLM(BaseVLM):
    model_id = "microsoft/kosmos-2-patch14-224"

    def _get_torch_dtype(self):
        mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        return mapping.get(self.config.dtype, torch.float16)

    def load(self) -> None:
        print(f"Loading {self.model_id}...")
        self.device = "cuda" if self.config.device == "auto" and torch.cuda.is_available() else self.config.device
        dtype = self._get_torch_dtype()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.model.eval()
        print(f"{self.model_id} loaded.")

    def preprocess(self, image: Image.Image, prompt: str) -> Any:
        # ВАЖНО: Kosmos-2 требует префикс <grounding> для поиска объектов
        # Если промпт пустой, просим описать изображение с координатами
        final_prompt = f"<grounding> {prompt if prompt else 'Describe this image'}"
        
        inputs = self.processor(text=final_prompt, images=image, return_tensors="pt")

        # Переносим все тензоры на девайс
        inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
        
        # В Kosmos-2 основной тензор изображений — pixel_values
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._get_torch_dtype())

        return inputs

    def generate(self, inputs: Any, max_new_tokens: int = 128, **kwargs) -> str:
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2 # Предотвращает зацикливание
            )
            
        # Декодируем, НЕ пропуская специальные токены (они нужны для координат)
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return generated_text

    def parse_coordinates(self, raw_output: str) -> List[Detection]:
        # В Kosmos-2 процессор имеет встроенный метод для извлечения боксов
        # Он парсит текст и возвращает найденные сущности и их координаты
        _, entities = self.processor.post_process_generation(raw_output)
        
        detections = []
        for label, (start, end), boxes in entities:
            for box in boxes:
                # box здесь уже в формате [x1, y1, x2, y2] и нормализован 0-1
                detections.append(Detection(
                    label=label,
                    bbox=list(box),
                    score=1.0
                ))
        return detections

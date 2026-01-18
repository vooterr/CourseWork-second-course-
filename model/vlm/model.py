import time
from typing import List

from model.vlm.abstract_model import BaseVLM
from model.vlm.quant import build_load_kwargs
from metrics.classification import (
    classification_accuracy,
    macro_f1
)
from utils.classification import normalize_prediction

from transformers import LlavaForConditionalGeneration, AutoProcessor, CLIPProcessor, CLIPModel
from PIL import Image
import torch

class LlavaVLM(BaseVLM):
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor_id =  "openai/clip-vit-large-patch14-336"


    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            use_fast=False
        )

        load_kwargs = build_load_kwargs(self.config)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            **load_kwargs
        )

        print(f"Model {self.model_id} loaded")

        self.model.eval()

    def preprocess(self, image, prompt: str):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=chat_prompt,
            return_tensors="pt"
        ).to(0, torch.float16)

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        return inputs
    

    def generate(self, inputs, max_new_tokens: int = 128) -> str:
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        return self.processor.decode(
            outputs[0][2:],
            skip_special_tokens=True
        )
    
    def evaluate_classification(self, dataset) -> dict:
        preds = []
        labels = []
        latencies = []

        for sample in dataset:
            image = sample["image"]
            prompt = sample["prompt"]
            classes = sample["classes"]
            label = sample["label"]

            full_prompt = (
                "Identify the dog breed shown in the image."
                "Output format: ONE WORD ONLY."
                "No explanation. No punctuation."
            )

            inputs = self.preprocess(image, full_prompt)

            start = time.time()
            output = self.generate(inputs)
            end = time.time()

            pred = normalize_prediction(output, classes)

            preds.append(pred)
            labels.append(label)
            latencies.append(end - start)

        return {
            "accuracy": classification_accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels),
            "avg_latency": sum(latencies) / len(latencies)
        }
        

class CLIPVLM(BaseVLM):
    model_id = "openai/clip-vit-large-patch14"
    
    
    def load(self):
        self.processor = CLIPProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        )
        
        self.model = CLIPModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir
        ).to(self.config.device)
        
        print(f"Model {self.model_id} loaded")
        
        self.model.eval()
        
        
    def preprocess(self, image, text_prompts: list):
        inputs = self.processor(
            text=text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        return inputs;


    def generate(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).squeeze(0)
        probs = similarity.softmax(dim=0)
        pred_id = probs.argmax().item()
        return pred_id, probs
    
    
    def generate_prompt(self, classes):
        classes = [' '.join(cls.split('_')) for cls in classes]
        text_prompts = [f"a photo of a {c}" for c in classes]
        return text_prompts
    

class GLIPVLM(BaseVLM):
    
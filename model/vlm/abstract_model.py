from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Optional, Literal
from dataclasses import dataclass


@dataclass
class ModelConfig:
    device: str = Literal["auto", "cuda", "cpu"]
    dtype: str = Literal["float16", "bfloat16", "float32"]
    quantization: Optional[str] = None


class BaseVLM(ABC):
    cache_dir = "C:/Users/ziglz/projects/coursework/models" 
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.processor = None
        print("Model ID: {MODEL_ID}")
    

    @abstractmethod
    def load(self) -> None:
        """Загрузка модели и процессора"""


    @abstractmethod
    def preprocess(self, image: Image.Image, prompt: str) -> Any:
        """Подготовка входных данных"""
        pass

    
    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> str:
        """Генерация ответа"""
        pass


    # @abstractmethod
    # def evaluate(self, dataset) -> dict:
    #     """
    #     Оценка модели на датасете.
    #     Возвращает словарь метрик.
    #     """
    #     pass
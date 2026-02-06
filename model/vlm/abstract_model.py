from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass

@dataclass
class Detection:
    label: str
    bbox: List[float] # [x1, y1, x2, y2]
    score: Optional[float] = None

@dataclass
class ModelConfig:
    device: str = Literal["auto", "cuda", "cpu"]
    dtype: str = Literal["float16", "bfloat16", "float32"]
    quantization: Optional[str] = None


class BaseVLM(ABC):
    cache_dir = "./models" 
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

    
    def parse_coordinates(self, raw_output: Any) -> List[Detection]:
        pass
    

    def detect(self, image: Image.Image, prompt: str = "Detect objects") -> List[Detection]:
        """
        Высокоуровневая обертка (Wrapper)
        """
        # 1. Подготавливаем данные
        inputs = self.preprocess(image, prompt)
        
        # 2. Получаем сырой ответ от модели (текст или тензор)
        raw_output = self.generate(inputs)
        
        # 3. Парсим результат в единый формат {label, bbox}
        # У каждой модели этот метод будет свой (post_process)
        formatted_output = self.parse_coordinates(raw_output)
        
        return formatted_output

    # @abstractmethod
    # def evaluate(self, dataset) -> dict:
    #     """
    #     Оценка модели на датасете.
    #     Возвращает словарь метрик.
    #     """
    #     pass

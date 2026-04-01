from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Optional, Literal, Any
from dataclasses import dataclass


@dataclass
class Detection:
    image_name: Optional[str]
    label: str
    bbox: List[float]  # [x1, y1, x2, y2]
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

    @abstractmethod
    def load(self) -> None:
        """Загрузка модели и процессора"""
        pass

    @abstractmethod
    def detect(self, image: Image.Image, classes: Optional[List[str]] = None) -> List[Detection]:
        """
        Единая точка входа для детекции.
        :param image: Входное изображение
        :param classes: Список классов для поиска (например, ["cat", "dog"]).
                        Если None, модель должна попытаться найти все объекты (если поддерживает).
        """
        pass

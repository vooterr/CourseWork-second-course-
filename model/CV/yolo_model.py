from ultralytics import YOLO
from PIL import Image
import torch

from model.vlm.abstract_model import Detection

import logging


class yolo_model:
    model_id: str
    device: str

    def __init__(self, model_id='yolo11n.pt', device="cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None


    def load(self):
        self.model = YOLO(self.model_id).to(torch.device(self.device if torch.cuda.is_available() else "cpu"))
        print(f"Loading {self.model_id}")
        logging.getLogger("ultralytics").setLevel(logging.ERROR)


    def detect(self, image: Image) -> list[Detection]:
        if self.model is None:
            raise KeyError("Download model, before detection")
        results = self.model(image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                conf_score = float(box.conf.item())

                # Получаем текстовое имя, если нужно
                label_name = self.model.names[class_id]

                detections.append(Detection(
                    label=label_name, # или class_id, зависит от твоего класса Detection
                    bbox=coords,
                    score=conf_score
                ))
        return detections
    def detect(self, path_to_dir: str) -> list[Detection]:
        if self.model is None:
            raise KeyError("Download model, before detection")
        results = self.model.predict(
            source=path_to_dir,
            device=torch.device('cuda') if torch.cuda.is_available()
                                else torch.device('cpu'),
        )
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                conf_score = float(box.conf.item())

                # Получаем текстовое имя, если нужно
                label_name = self.model.names[class_id]

                detections.append(Detection(
                    image_name=r.path[r.path.rfind('/') + 1:],
                    label=label_name, # или class_id, зависит от твоего класса Detection
                    bbox=coords,
                    score=conf_score
                ))
        return detections

    def detect(self, imgs) -> list[Detection]:
        if self.model is None:
            raise KeyError("Download model, before detection")
        results = self.model.predict(
            source=imgs,
            device=torch.device('cuda') if torch.cuda.is_available()
                                else torch.device('cpu'),
        )
        detections = []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                class_id = int(box.cls.item())
                conf_score = float(box.conf.item())


                label_name = self.model.names[class_id]

                detections.append(Detection(
                    image_name=r.path,
                    label=label_name,
                    bbox=coords,
                    score=conf_score
                ))
        return detections

if __name__ == "__main__":
    model = yolo_model()
    model.load()
    img = Image.open('data/sign_detection/images/road0.png')
    model.detect([img])

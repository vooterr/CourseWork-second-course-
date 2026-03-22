from ultralytics import YOLO
from PIL import Image

from model.vlm.abstract_model import Detection


class yolo_model:
    
    
    def __init__(self):
        self.model = None
    
    
    def load(self, model_id: str = 'yolo11n.pt'):
        self.model = YOLO(model_id)
        print(f"Loading {model_id}")
    
    
    def detect(self, image) -> list[Detection]:
        if self.model is None:
            raise KeyError("Download model, before detection")
        results = self.model(image, )
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

if __name__ == "__main__":
    model = yolo_model()
    model.load()
    img = Image.open('data/sign_detection/images/road0.png')
    model.detect(img)
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch

from model.CV.yolo_model import yolo_model
from metrics.detection import map_to_str, map_coco

def evaluate_model_per_img(
    limit: int | None = None,
    model=None,
    input_dir: str = 'data/coco128',
    output_file: str = 'predictions',
    classes: list[str] | None = None
): 
    if model is None:
        return pd.DataFrame(columns=['image_name', 'class', 'x1', 'y1', 'x2', 'y2'])
    
    model.load()
    img_dir = Path(input_dir)
    rows_all = []
    
    
    for idx, img_path in tqdm(enumerate(sorted(img_dir.iterdir()))):
        if limit != None and idx >= limit: 
            break
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        img = Image.open(img_path).convert('RGB')

        dets = model.detect(img)
        for d in dets:
            rows_all.append(
                {
                    "image_name": img_path.name,
                    "class": d.label,
                    "score": float(d.score),
                    "x1": float(d.bbox[0]),
                    "y1": float(d.bbox[1]),
                    "x2": float(d.bbox[2]),
                    "y2": float(d.bbox[3])
                }
            )
        if classes:
            pass#TODO
    pred = pd.DataFrame(data=rows_all, columns=['image_name', 'class', 'score', 'x1', 'y1', 'x2', 'y2'])
    pred.to_csv(output_file, index=False)
    return pred


if __name__ == "__main__":
    model = yolo_model()
    
    input_dir = "data/coco/coco2017/train2017"
    ann_file = 'data/coco/coco2017/annotations/annotations_train2017.csv'
    output_dir = "predictions/coco_dino.csv"
    
    pred = evaluate_model_per_img(limit=2000, model=model, input_dir=input_dir, output_file=output_dir)
    gt = pd.read_csv(
        ann_file
    )
    
    classes = sorted(pd.unique(gt["class"]).tolist())
    
    processed_images = pred["image_name"].unique()
    gt_subset = gt[gt["image_name"].isin(processed_images)].copy()
    
    print(map_to_str(map_coco(pred=pred, gt=gt_subset, classes=classes)))

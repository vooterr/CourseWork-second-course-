from model.vlm.model import Florence2VLM, QwenVLM, DinoVLM
from model.vlm.abstract_model import ModelConfig
from metrics.detection import map_coco, match_classes, map_to_str

from tqdm import tqdm

import pandas as pd
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ====== 1) нормализация ======
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    return df

def evaluate_model(
    limit: int = 100,
    model=None,
    input_dir: str = "data/coco128",
    output_file: str = "pred.csv",
    classes: list[str] | None = None,
) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame(columns=["image_name", "class", "score", "x1", "y1", "x2", "y2"])

    model.load()
    img_dir = Path(input_dir)

    rows_all = []
    for idx, img_path in enumerate(sorted(img_dir.iterdir())):
        if idx >= limit:
            break
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        img = Image.open(img_path).convert("RGB")
        dets = model.detect(img, classes)  # твой интерфейс

        for d in dets:
            rows_all.append({
                "image_name": img_path.name,
                "class": d.label,
                "score": float(d.score),
                "x1": float(d.bbox[0]),
                "y1": float(d.bbox[1]),
                "x2": float(d.bbox[2]),
                "y2": float(d.bbox[3]),
            })

    pred = pd.DataFrame(rows_all, columns=["image_name", "class", "score", "x1", "y1", "x2", "y2"])
    pred.to_csv(output_file, index=False)
    return pred


def evaluate_model_per_img(
    limit: int | None = None,
    model=None,
    input_dir: str = 'data/coco128',
    ann_file: str = 'data/coco128/annotations.csv',
    output_file: str = 'predictions',
    classes: list[str] | None = None
): 
    if model is None:
        return pd.DataFrame(columns=['image_name', 'class', 'score', 'x1', 'y1', 'x2', 'y2'])
    
    model.load()
    img_dir = Path(input_dir)
    rows_all = []
    
    ann = pd.read_csv(ann_file)
    ann_grp = ann.groupby('image_name').agg(list)['class']
    pd.DataFrame(columns=['image_name', 'class', 'score', 'x1', 'y1', 'x2', 'y2']).to_csv(output_file, index=False)
    for idx, img_path in tqdm(enumerate(sorted(img_dir.iterdir()))):
        if limit != None and idx >= limit: 
            break
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
            continue
        
        img = Image.open(img_path).convert('RGB')
        if (img_path.name not in ann_grp.index):
            continue
        
        clss = list(set(ann_grp.loc[img_path.name]))
        dets = model.detect(img, clss)
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
            
        if (idx % 1000 == 0 and idx != 0):
            temp_df = pd.DataFrame(rows_all)
            temp_df.to_csv(output_file, mode='a', index=False, header=False)


    pd.DataFrame(rows_all).to_csv(output_file, mode='a', index=False, header=False)
    pred = pd.read_csv(output_file)
    return pred
        


# ====== 4) main-проверка ======
if __name__ == "__main__":
    config = ModelConfig(device='auto', dtype='float16')
    model = DinoVLM(config)

    input_dir = "data/coco/coco2017/train2017"
    ann_file = 'data/coco/coco2017/annotations/annotations_train2017.csv'
    output_dir = "predictions/coco_dino_tiny.csv"

    gt = pd.read_csv(
        ann_file
    )

    gt = normalize_df(gt)
    classes = sorted(pd.unique(gt["class"]).tolist())

    pred = evaluate_model_per_img(model=model, input_dir=input_dir, ann_file=ann_file, output_file=output_dir, classes=classes)
    pred = normalize_df(pred)

    processed_images = pred["image_name"].unique()
    gt_subset = gt[gt["image_name"].isin(processed_images)].copy()

    print("raw pred classes:\n", pred["class"].value_counts(dropna=False))

    print("reduced pred classes:\n", pred["class"].value_counts(dropna=False))

    print("pred rows:", len(pred), "gt rows:", len(gt))
    print("common classes:", sorted(set(pred["class"]) & set(gt["class"])))
        
    print(map_to_str(map_coco(pred=pred, gt=gt_subset, classes=classes)))
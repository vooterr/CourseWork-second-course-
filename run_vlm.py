from model.vlm.model import Florence2VLM, QwenVLM, DinoVLM
from model.vlm.abstract_model import ModelConfig
from metrics.detection import map_coco, match_classes, map_to_str

from tqdm import tqdm
import argparse

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
            rows_all = []
            temp_df.to_csv(output_file, mode='a', index=False, header=False)


    pd.DataFrame(rows_all).to_csv(output_file, mode='a', index=False, header=False)
    pred = pd.read_csv(output_file)
    return pred
        


# ====== 4) main-проверка ======
import argparse

# ... (остальной ваш код импортов и функций) ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of VLM models")
    parser.add_argument("--input_dir", type=str, required=True, help="Путь к папке с изображениями")
    parser.add_argument("--ann_file", type=str, required=True, help="Путь к файлу аннотаций")
    parser.add_argument("--output_file", type=str, required=True, help="Куда сохранить predictions.csv")
    parser.add_argument("--model", type=str, default="dino", choices=["dino", "florence", "qwen"], help="Выбор модели")
    parser.add_argument("--limit", type=int, default=None, help="Лимит изображений для теста")
    
    args = parser.parse_args()

    config = ModelConfig(device='auto', dtype='float16')
    
    # Инициализация нужной модели
    model = DinoVLM(config=config)

    # Загрузка GT
    gt = pd.read_csv(args.ann_file)
    gt = normalize_df(gt)
    classes = sorted(pd.unique(gt["class"]).tolist())

    # Запуск
    print(f"Запуск модели {args.model} на данных {args.input_dir}...")
    pred = evaluate_model_per_img(
        limit=args.limit,
        model=model, 
        input_dir=args.input_dir, 
        ann_file=args.ann_file, 
        output_file=args.output_file, 
        classes=classes
    )
    
    pred = normalize_df(pred)
    processed_images = pred["image_name"].unique()
    gt_subset = gt[gt["image_name"].isin(processed_images)].copy()

    print("raw pred classes:\n", pred["class"].value_counts(dropna=False))
    print("pred rows:", len(pred), "gt rows:", len(gt))
    
    # Расчет метрик
    print(map_to_str(map_coco(pred=pred, gt=gt_subset, classes=classes)))
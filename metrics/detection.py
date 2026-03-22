import json
import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from tqdm import tqdm # Рекомендую добавить для прогресс-бара
from pathlib import Path
import pandas as pd
import numpy as np

# Импортируем типы из твоего файла с классами
# Предполагаем, что они лежат в model.vlm.base (поправь импорт под свою структуру)
from model.vlm.abstract_model import BaseVLM, Detection

# def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
#     """
#     Вычисляет Intersection over Union (IoU) между двумя боксами.
#     Format: [x1, y1, x2, y2]
#     """
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
#     return iou

def match_boxes(
    preds: List[Detection], 
    gts: List[Detection], 
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Сопоставляет предсказания и истинные значения.
    Возвращает (TP, FP, FN)
    """
    # 1. Группируем по классам, так как сравнивать кота с собакой нет смысла
    # Но если задача class-agnostic (просто найти объекты), этот шаг можно пропустить.
    # Здесь делаем строгое сравнение по label.
    
    tp, fp = 0, 0
    # Создаем копии списков, чтобы удалять найденные
    gts_pool = gts.copy()
    preds_pool = preds.copy()

    # Сначала ищем совпадения
    matches = []
    
    for p_idx, p in enumerate(preds_pool):
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx, g in enumerate(gts_pool):
            # Сверяем лейбл (case-insensitive)
            if p.label.lower() != g.label.lower():
                continue
            
            iou = calculate_iou(p.bbox, g.bbox)
            if iou >= best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= iou_threshold:
            matches.append((p_idx, best_gt_idx, best_iou))

    # Сортируем совпадения по IoU (Greedy matching - забираем лучшие первыми)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for p_idx, g_idx, _ in matches:
        if g_idx not in matched_gt_indices and p_idx not in matched_pred_indices:
            tp += 1
            matched_gt_indices.add(g_idx)
            matched_pred_indices.add(p_idx)

    # FP = Предсказания, которым не хватило пары
    fp = len(preds) - tp
    
    # FN = Истинные объекты, которые не нашли
    fn = len(gts) - tp
    
    return tp, fp, fn

def compute_image_metrics(
    model: BaseVLM, 
    image_path: str, 
    ground_truths: List[Dict], 
    prompt: str = "Detect objects"
) -> Dict[str, int]:
    """
    Обрабатывает ОДНО изображение:
    1. Загружает картинку
    2. Запускает модель
    3. Парсит GT из формата датасета в формат Detection
    4. Считает метрики
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return {"tp": 0, "fp": 0, "fn": 0}

    # 1. Инференс
    predictions = model.detect(image, prompt=prompt)
    
    # 2. Подготовка GT (предполагаем формат input dict: {'label': 'dog', 'bbox': [x1, y1, x2, y2]})
    gt_detections = [
        Detection(label=item['label'], bbox=item['bbox']) 
        for item in ground_truths
    ]

    # 3. Расчет
    tp, fp, fn = match_boxes(predictions, gt_detections, iou_threshold=0.5)
    
    return {"tp": tp, "fp": fp, "fn": fn}

def evaluate_dataset(
    model: BaseVLM, 
    dataset_dir: str, 
    annotation_file: str,
    prompt: str = "Detect objects"
) -> Dict[str, float]:
    """
    Главная функция оценки.
    dataset_dir: путь к папке с изображениями
    annotation_file: путь к json файлу вида [{"file_name": "1.jpg", "objects": [...]}]
    """
    
    # Загружаем аннотации
    with open(annotation_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_tp, total_fp, total_fn = 0, 0, 0
    
    print(f"Starting evaluation on {len(dataset)} images...")

    for item in tqdm(dataset):
        file_name = item['file_name']
        image_path = os.path.join(dataset_dir, file_name)
        gt_objects = item.get('objects', []) # Expects list of dicts with label and bbox
        
        metrics = compute_image_metrics(model, image_path, gt_objects, prompt)
        
        total_tp += metrics['tp']
        total_fp += metrics['fp']
        total_fn += metrics['fn']

    # Финальные метрики
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "raw_counts": {"tp": total_tp, "fp": total_fp, "fn": total_fn}
    }
    
    return results


def calculate_iou(
    a: List[float] | np.ndarray,
    b: list[float] | np.ndarray,
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    
    inter = iw * ih
    area1 = max(0, (a[2] - a[0]) * (a[3] - a[1]))
    area2 = max(0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def match_classes(
    pred_cls: pd.DataFrame | None = None,
    gt_cls: pd.DataFrame | None = None,
    iou_trh: float = 0.5
):
    BOX_COLS = ['x1', 'y1', 'x2', 'y2']
    if (pred_cls is None or gt_cls is None):
        return 
    pred_cls = pred_cls.sort_values("score", ascending=False).reset_index(drop=True)
    
    gt_by_img = {img: g.reset_index(drop=True) for img, g in gt_cls.groupby("image_name")}
    used_by_img = {img: np.zeros(len(g), dtype=bool) for img, g in gt_by_img.items()}

    tp = np.zeros(len(pred_cls), dtype=np.int8)
    fp = np.zeros(len(pred_cls), dtype=np.int8)
    
    for i, p in pred_cls.iterrows():
        i = int(i)
        img = p['image_name']
        if img not in gt_by_img:
            fp[i] = 1
            continue
        
        g = gt_by_img[img]
        used = used_by_img[img]
        
        p_box = p[BOX_COLS].to_numpy(dtype=float)
        
        best_iou = 0
        best_j = -1
        
        for j in range(len(g)):
            if used[j]:
                continue
            g_box = g.loc[j, BOX_COLS].to_numpy(dtype=float)
            v = calculate_iou(g_box, p_box)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou > iou_trh and best_j != -1:
            tp[i] = 1
            used[best_j] = True
        else:
            fp[i] = 1
    scores = pred_cls['score'].to_numpy()
    n_gt = len(gt_cls)
    return tp, fp, scores, n_gt
        

def precision_recall(tp, fp, n_gt):
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    denom = np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    precision = tp_cumsum / denom
    recall = tp_cumsum / max(int(n_gt), 1)
    
    return precision, recall
    

def ap_from_pr(prec: np.ndarray, rec: np.ndarray, n_gt: int | None = None) -> float:
    if n_gt is not None and n_gt == 0:
        return float("nan")

    # если предсказаний нет, а GT есть -> AP = 0
    if prec is None or rec is None or len(prec) == 0 or len(rec) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_model(
    limit: int = 100,
    model: BaseVLM | None = None,
    input_dir: str = 'data/coco128',
    ann_file: str = 'annotation.csv',
    classes: List[str] = []
):
    if not model:
        return
    
    model.load()
    
    img_dir = Path(input_dir)
    
    df = pd.DataFrame(columns=["image_name", "class", "score", "x1", "y1", "x2", "y2"])
    
    for idx, img_path in enumerate(img_dir.iterdir()):
        if idx > limit:
            break
        img = Image.open(img_path)
        
        detections = model.detect(img, classes)
        rows = [
            {"image_name": img_path.name,
                "class": detect.label,
                "score": detect.score,
                "x1": detect.bbox[0],
                "y1": detect.bbox[1],
                "x2": detect.bbox[2],
                'y2': detect.bbox[3]
            } 
            for detect in detections
        ]
        #print(rows)
        if rows:
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        
    df.to_csv(ann_file, index=False)
    return df
    
    
def map_voc(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    iou_thr: float = 0.5,
    classes: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    if classes is None:
        classes = sorted(set(gt["class"].unique()) | set(pred["class"].unique()))

    ap_by_class: dict[str, float] = {}

    for cls in classes:
        pred_cls = pred[pred["class"] == cls].copy()
        gt_cls = gt[gt["class"] == cls].copy()

        if len(gt_cls) == 0:
            ap_by_class[cls] = np.nan  # или 0.0, выбери политику
            continue

        tp, fp, scores, n_gt = match_classes(pred_cls, gt_cls, iou_trh=iou_thr)
        prec, rec = precision_recall(tp, fp, n_gt)
        ap_by_class[cls] = ap_from_pr(prec, rec)

    vals = np.array(list(ap_by_class.values()), dtype=float)
    vals = vals[~np.isnan(vals)]
    mAP = float(np.mean(vals)) if len(vals) else float("nan")
    return mAP, ap_by_class
    
    
def map_coco(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    iou_thrs: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    classes: list[str] | None = None,
) -> dict:
    if classes is None:
        classes = sorted(set(gt["class"].unique()) | set(pred["class"].unique()))

    ap_per_iou = {}
    ap_per_class = {cls: [] for cls in classes}

    for t in iou_thrs:
        mAP_t, ap_by_class_t = map_voc(pred, gt, iou_thr=t, classes=classes)
        ap_per_iou[t] = mAP_t
        for cls in classes:
            ap_per_class[cls].append(ap_by_class_t.get(cls, np.nan))

    mAP = float(np.nanmean(list(ap_per_iou.values())))
    return {
        "mAP": mAP,
        "mAP_per_iou": ap_per_iou,
        "AP_matrix_per_class": ap_per_class,  # список AP по IoU для каждого класса
    }
    
def map_to_str(
    map: dict,
    iou_thrs: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> str:

        
    output = f'mAP: {float(map['mAP']):<8.4f}\n'
    output += 'mAP per IoU\n:'
    
    for thr in iou_thrs:
        output += f'\tmAP@{thr}: {map['mAP_per_iou'][thr]:.4f}\n'
    output += "AP per cls\n"
    df = pd.DataFrame(map['AP_matrix_per_class']).T
    df.columns = iou_thrs
    df.index.name = 'Class'
    output += df.to_string()
        
    return output
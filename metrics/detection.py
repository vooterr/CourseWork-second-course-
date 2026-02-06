import json
import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from tqdm import tqdm # Рекомендую добавить для прогресс-бара

# Импортируем типы из твоего файла с классами
# Предполагаем, что они лежат в model.vlm.base (поправь импорт под свою структуру)
from model.vlm.abstract_model import BaseVLM, Detection

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Вычисляет Intersection over Union (IoU) между двумя боксами.
    Format: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

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
            if iou > best_iou:
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

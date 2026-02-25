import json
import os
import time
from tqdm import tqdm
from PIL import Image

# Импорты
from model.vlm.abstract_model import ModelConfig, Detection
from model.vlm.model import DinoVLM, KosmosVLM, Florence2VLM 
from metrics.detection import match_boxes
from utils.visualizer import draw_and_save 

# --- НАСТРОЙКИ ---
IMAGES_DIR = "data/coco/val2017"
ANNOTATION_FILE = "data/coco/annotations/instances_val2017.json"
OUTPUT_DIR = "predictions_kosmos"
LIMIT = 1000  # Количество картинок (None = все)

def load_coco_data(json_path):
    """Загрузка аннотаций COCO и подготовка датасета."""
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Словари для быстрого маппинга
    categories = {c['id']: c['name'] for c in data['categories']}
    images = {img['id']: img['file_name'] for img in data['images']}
    
    # Группировка аннотаций по ID изображения
    img_annotations = {}
    for ann in data['annotations']:
        if ann.get('iscrowd'): continue
        
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        
        # Конвертация bbox из [x, y, w, h] в [x1, y1, x2, y2]
        x, y, w, h = ann['bbox']
        label = categories.get(ann['category_id'], "unknown")
        
        # Сразу создаем объект Detection для GT
        img_annotations[img_id].append(Detection(
            label=label, 
            bbox=[x, y, x + w, y + h]
        ))

    # Формирование итогового списка
    dataset = []
    for img_id, file_name in images.items():
        if img_id in img_annotations:
            gt_objects = img_annotations[img_id]
            dataset.append({
                "file_name": file_name,
                "ground_truth": gt_objects,
                # Список уникальных классов для поиска на этом изображении
                "classes": list(set(obj.label for obj in gt_objects))
            })
            
    return dataset

def main(output_dir: str, model):
    # 1. Инициализация
    model.load()
    
    # 2. Подготовка данных
    dataset = load_coco_data(ANNOTATION_FILE)
    if LIMIT: 
        dataset = dataset[:LIMIT]
        print(f"⚠️ Limit set: processing {LIMIT} images")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Счетчики метрик
    total_tp, total_fp, total_fn = 0, 0, 0
    total_time = 0
    
    print("🚀 Starting validation loop...")
    
    for item in tqdm(dataset):
        img_path = os.path.join(IMAGES_DIR, item['file_name'])
        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            target_classes = item['classes']
            ground_truth = item['ground_truth']

            # 3. Инференс
            start_time = time.time()
            
            # ВАЖНО: Теперь передаем список классов, модель сама сформирует промпт
            predictions = model.detect(image, classes=target_classes)
            
            total_time += (time.time() - start_time)

            # 4. Метрики (сравнение предсказаний с GT)
            # Функция match_boxes должна возвращать (TP, FP, FN) для одного изображения
            tp, fp, fn = match_boxes(predictions, ground_truth, iou_threshold=0.5)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # 5. Визуализация
            draw_and_save(image, ground_truth, predictions, item['file_name'], OUTPUT_DIR)

        except Exception as e:
            print(f"Error processing {item['file_name']}: {e}")

    # 6. Итоговый отчет
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    avg_fps = len(dataset) / total_time if total_time > 0 else 0

    print("\n" + "="*30)
    print(f" RESULTS ({len(dataset)} images):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg FPS:   {avg_fps:.2f}")
    print(f"Raw: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print("="*30)

if __name__ == "__main__":
    
    config = ModelConfig(device="auto", dtype="float16")
    # model = Florence2VLM(config)
    # main(output_dir='predictions/florence2', model=model)
    model = DinoVLM(config)
    main(output_dir="predictions/dino", model=model)

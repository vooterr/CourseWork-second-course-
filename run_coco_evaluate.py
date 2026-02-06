import json
import os
import time
from tqdm import tqdm
from PIL import Image

# Импорты (убедись, что пути корректны)
from model.vlm.model import Florence2VLM, KosmosVLM, ModelConfig
from metrics.detection import Detection, match_boxes
from utils.visualizer import draw_and_save 

# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ---
IMAGES_DIR = "data/coco/val2017"
ANNOTATION_FILE = "data/coco/annotations/instances_val2017.json"
OUTPUT_DIR = "predictions_vis"  # Куда сохранять картинки с рамками
LIMIT = 50                      # Количество картинок для теста (None = все)
SAVE_VISUALIZATION = True       # Рисовать ли рамки

target = set()

def load_coco_data(json_path):
    """Быстрый парсинг COCO аннотаций"""
    print(f"📂 Loading annotations from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Маппинги
    id_to_name = {c['id']: c['name'] for c in data['categories']}
    id_to_file = {i['id']: i['file_name'] for i in data['images']}
    
    # Сборка датасета
    dataset = []
    # Группируем аннотации по image_id через временный словарь
    img_anns = {}
    for ann in data['annotations']:
        if ann.get('iscrowd'): continue
        img_id = ann['image_id']
        if img_id not in img_anns: img_anns[img_id] = []
        
        # COCO [x,y,w,h] -> [x1,y1,x2,y2]
        x, y, w, h = ann['bbox']
        img_anns[img_id].append({
            "label": id_to_name[ann['category_id']],
            "bbox": [x, y, x + w, y + h]
        })
        target.add(id_to_name[ann['category_id']])

    for img_id, objs in img_anns.items():
        if img_id in id_to_file:
            dataset.append({"file_name": id_to_file[img_id], "objects": objs})
            
    return dataset, target

def main():
    # 1. Инициализация
    config = ModelConfig(device="auto", dtype="float16")
    model = Florence2VLM(config)

    model.load()
    
    # 2. Данные
    dataset, target = load_coco_data(ANNOTATION_FILE)
    if LIMIT: 
        dataset = dataset[:LIMIT]
        print(f"⚠️ Limit set: processing {LIMIT} images")

    # 3. Основной цикл (Инференс + Метрики + Визуал)
    total_tp, total_fp, total_fn = 0, 0, 0
    total_time = 0
    
    print(f"🚀 Starting processing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target = ", ".join(list(target))
    prompt = f"<OPEN_VOCABULARY_DETECTION>"    
    for item in tqdm(dataset):
        img_path = os.path.join(IMAGES_DIR, item['file_name'])
        if not os.path.exists(img_path): continue

        try:
            image = Image.open(img_path).convert("RGB")
            gts = [Detection(o['label'], o['bbox']) for o in item['objects']]

            target = [o['label'] for o in item["objects"]]
            preds = []
            start = time.time()

            for tar in target:

            
                prompt_with_cls = f"{prompt}object: {tar}"
            # A. Инференс
                start = time.time()
                detect = model.detect(image, prompt=prompt_with_cls)
                for dec in detect:
                    dec.label = dec.label.replace("object: ", "")
                preds += detect
            total_time += (time.time() - start)

            
            # B. Подготовка GT
            gts = [Detection(o['label'], o['bbox']) for o in item['objects']]

            # C. Метрики (считаем match для текущего кадра)
            tp, fp, fn = match_boxes(preds, gts, iou_threshold=0.5)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # D. Визуализация и сохранение
            if SAVE_VISUALIZATION:
                draw_and_save(image, gts, preds, item['file_name'], OUTPUT_DIR)

        except Exception as e:
            print(f"Error processing {item['file_name']}: {e}")

    # 4. Итоговый отчет
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    avg_time = total_time / len(dataset) if dataset else 0

    print("\n" + "="*30)
    print(f"📊 RESULTS on {len(dataset)} images:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Avg Time:  {avg_time:.4f}s")
    print(f"Raw: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    main()

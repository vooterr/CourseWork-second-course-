import os
from PIL import Image, ImageDraw, ImageFont
from typing import List
from model.vlm.abstract_model import Detection

def draw_and_save(
    image: Image.Image,
    ground_truths: List[Detection],
    predictions: List[Detection],
    filename: str,
    output_dir: str = "predictions"
):
    """
    Рисует GT (зеленым) и Preds (красным) на изображении и сохраняет его.
    """
    # Создаем копию, чтобы не портить оригинал в памяти
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Попробуем загрузить шрифт, иначе используем дефолтный
    try:
        # Для Windows путь может отличаться, например arial.ttf
        # Для Linux часто DejaVuSans.ttf
        font = ImageFont.load_default() 
    except:
        font = None

    # --- Рисуем Ground Truth (ЗЕЛЕНЫЕ) ---
    for gt in ground_truths:
        # Рисуем рамку (width=3 для жирности)
        draw.rectangle(gt.bbox, outline="lime", width=3)
        # Рисуем подложку для текста
        text_origin = (gt.bbox[0], max(0, gt.bbox[1] - 15))
        draw.text(text_origin, f"GT: {gt.label}", fill="lime", font=font)

    # --- Рисуем Predictions (КРАСНЫЕ) ---
    for pred in predictions:
        # Слегка смещаем рамку, чтобы они не перекрывали GT идеально
        box = [c + 2 for c in pred.bbox] 
        draw.rectangle(box, outline="red", width=2)
        
        # Текст предсказания (снизу бокса или чуть правее)
        text_origin = (box[0], box[1])
        label_text = f"{pred.label} ({pred.score:.2f})" if pred.score else pred.label
        
        # Черная обводка для текста, чтобы читалось на любом фоне
        draw.text(text_origin, label_text, fill="red", font=font)

    # --- Сохранение ---
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    vis_image.save(save_path)
    # print(f"Saved visualization to {save_path}")

from pathlib import Path
from tqdm import tqdm

import fiftyone as fo
import pandas as pd
import argparse
from PIL import Image

def make_dataset(
    predictions_file: str,
    annotatin_file: str,
    img_dir: str,
    dataset_name: str
) -> fo.Dataset:
    dataset = fo.Dataset()
    img_dir_path = Path(img_dir)
    
    data = pd.read_csv(annotatin_file)
    predictions = pd.read_csv(predictions_file)
    
    data_grp = data.groupby('image_name').agg(list)
    predictions_grp = predictions.groupby('image_name').agg(list)
    
    for img_path in tqdm(img_dir_path.iterdir()):
        img = Image.open(img_path)
        w, h = img.size
        sample = fo.Sample(img_path)
        img_name = img_path.name
        if (img_path.name not in data_grp.index):
            continue
        detect_per_file = data_grp.loc[img_name, :]
        fo_detections = []
        #print(detect_per_file)
        for cls, x1, y1, x2, y2 in zip(detect_per_file['class'], detect_per_file['x1'], detect_per_file['y1'], detect_per_file['x2'], detect_per_file['y2']):
            fo_detections.append(fo.Detection(
                label=cls,
                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            ))
        fo_predictions = []
        if (img_name in predictions_grp.index):
            predict_per_file = predictions_grp.loc[img_name, :]
            for cls, score, x1, y1, x2, y2 in zip(predict_per_file['class'], predict_per_file['score'], predict_per_file['x1'], predict_per_file['y1'], predict_per_file['x2'], predict_per_file['y2']):
                fo_predictions.append(fo.Detection(
                    label=cls,
                    bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    confidence=score
                ))
        
        sample['ground_truth'] = fo.Detections(detections=fo_detections)
        sample['predictions'] = fo.Detections(detections=fo_predictions)
        dataset.add_sample(sample)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This a fo evaluation dataset")
    parser.add_argument("ann_file", help="CSV файл с разметракми изображений")
    parser.add_argument("pred_file")
    parser.add_argument("img_dir")
    parser.add_argument("dataset_name")
    
    args = parser.parse_args()
    
    dataset = make_dataset(
        annotatin_file=args.ann_file,
        predictions_file=args.pred_file,
        img_dir=args.img_dir,
        dataset_name=args.dataset_name
    )
    
    session = fo.launch_app(dataset)
    session.wait()
    
#Было найдено большое количество галюцинаций в датасете с road_sign_detection на объектах speedlimit и crosswalk, возможно dino не особо понимает контекста 
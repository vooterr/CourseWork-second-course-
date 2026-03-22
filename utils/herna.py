import json
from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET

import pandas as pd


def transform_vehicle_data():
    with open("data/vehicle/No_Apply_Grayscale/No_Apply_Grayscale/Vehicles_Detection.v8i.coco/train/_annotations.coco.json", 'r') as f:
        data = json.load(f)

    cat_id2name = {c["id"]: c["name"] for c in data['categories']}
    img_id2info = {im['id']: im for im in data['images']}

    rows = []

    for ann in data['annotations']:
        im = img_id2info[ann['image_id']]
        rows.append({
            "image_id": ann["image_id"],
            "file_name": im["file_name"],
            "w": im["width"],
            "h": im["height"],
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "category": cat_id2name.get(ann["category_id"], f"unknown_{ann['category_id']}"),
            "x1": ann["bbox"][0],
            "y1": ann["bbox"][1],
            "x2": ann['bbox'][0] + ann["bbox"][2],
            "y2": ann['bbox'][1] + ann["bbox"][3],
        })
        
    print('images', len(data['images']))


    cnt = Counter(r['category'] for r in rows)
    print('top calses')

    for k, v in cnt.most_common(20):
        print(f"{k:15s} {v}")

    df = pd.DataFrame(rows, columns=["image_id", "file_name", "w", "h", "ann_id", "category_id", "category", "x1", "y1", "x2", "y2"])
    df = df.rename(columns={"category": "class", "file_name": "image_name"})
    df['class'] = df['class'].apply(lambda x: x.lower())
    print(df.head())
    df.to_csv("data/vehicle/No_Apply_Grayscale/No_Apply_Grayscale/Vehicles_Detection.v8i.coco/train/annotations.csv")

def convert_coco_data(
    input_file: str,
    output_file: str
):
    #input_file = 'data/coco/coco2017/annotations/instances_train2017.json'
    
    out_file = pd.DataFrame(
        columns=['image_name', 'class', 'x1', 'y1', 'x2', 'y2']
    )
    with open(input_file, 'r') as f:
        coco_file = json.load(f)
        
    cls_dict = {}
    for row in coco_file['categories']:
        cls = row['name']
        id = row['id']
        cls_dict[id] = cls
    
    images = coco_file['images']
    img_dict = {img['id']: img['file_name'] for img in images}
    #print(img_dict[558840])
    rows = []

    for row in coco_file['annotations']:
        #print(row)
        img_name = img_dict[row['image_id']]
        cls = cls_dict[row['category_id']]
        bbox = row['bbox']
        rows.append({
            'image_name': img_name,
            'class': cls,
            'x1': bbox[0],
            'y1': bbox[1],
            'x2': bbox[0] + bbox[2],
            'y2': bbox[1] + bbox[3]
        })
    out_file = pd.DataFrame(data=rows)
    out_file.to_csv(output_file)


def parse_xml_files(
    ann_dir: str,
    output_file: str
) -> None:
    
    ann_path = Path(ann_dir)
    rows = []
    for file in ann_path.iterdir():
        if file.suffix != '.xml':
            continue
        
        tree = ET.parse(file)
        f_name = tree.find('filename').text
        for obj in tree.findall('object'):
            cls = obj.find('name').text
            
            bnd = obj.find('bndbox')
            x1, y1, x2, y2 = (
                int(bnd.find('xmin').text), 
                int(bnd.find('ymin').text),
                int(bnd.find('xmax').text), 
                int(bnd.find('ymax').text)
            )
            rows.append({
                'image_name': f_name,
                'class': cls,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    
    
def download_data():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("andrewmvd/road-sign-detection")

    print("Path to dataset files:", path)


def transform_sign_dataset(
    ann_path: str
) -> None:
    df = pd.read_csv(ann_path)
    dict = {'trafficlight': 'trafficlight', 'stop': 'stop signs', 'speedlimit': 'speed limit signs', 'crosswalk': 'pedestrian crossing sign'}
    #df['class'] = df['class'].apply(lambda x: x.replace('sign ', ''))
    df['class'] = df['class'].map(dict)
    df.to_csv(ann_path)
    
    
if __name__ == "__main__":
    parse_xml_files('data/sign_detection/annotations', 'data/sign_detection/annotations/annotations.csv')
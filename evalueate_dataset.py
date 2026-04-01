from metrics.detection import map_coco, match_classes, map_to_str
import pandas as pd

def evaluate_dataset(gt_path, pred_path):
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)
    map_score = map_coco(gt, pred)
    return map_score

if __name__ == "__main__":
    pred_dino_path = 'predictions/predictions_dino_night.csv'
    pred_yolo_path = 'predictions/coco_yolo.csv'
    gt_path = 'data/coco/coco2017/annotations/annotations_train2017.csv'
    gt = pd.read_csv(gt_path)
    gt = gt.drop(columns=['idx'])

    pred_dino = pd.read_csv(pred_dino_path)
    pred_yolo = pd.read_csv(pred_yolo_path)
    pred_yolo.image_name = pred_yolo.image_name.apply(lambda x: x[x.rfind('/')+1:])

    print(pred_yolo.head(), pred_dino.head(), sep='\n')

    gt = gt[gt['image_name'].isin(pred_dino['image_name'])]
    gt = gt[gt['image_name'].isin(pred_yolo['image_name'])]

    pred_yolo = pred_yolo[pred_yolo['image_name'].isin(gt['image_name'])]

    clss = dict(gt['class'].value_counts().sort_values(ascending=False)).items()

    five_clss = sorted(clss, key=lambda x: x[1], reverse=True)[:5]
    five_clss = [x[0] for x in five_clss]

    gt = gt[gt['class'].isin(five_clss)]
    pred_dino = pred_dino[pred_dino['class'].isin(five_clss)]
    pred_yolo = pred_yolo[pred_yolo['class'].isin(five_clss)]

    # print(pred_dino.shape[0], gt.shape[0])
    # print(pred_yolo.shape[0], gt.shape[0])

    # print(map_coco(gt=gt, pred=pred, classes=five_clss))
    # with open('predictions/data_for_pred_dino_night.txt', 'r') as f:
    #     data = dict(eval(f.read()))
    # print(map_to_str(data))
    print(map_coco(gt=gt, pred=pred_yolo, classes=five_clss))

from pathlib import Path
import asyncio
from typing import Optional
import argparse

from PIL import Image
from model.vlm.abstract_model import Detection
from tqdm import tqdm
import pandas as pd
import torch


from model.CV.yolo_model import yolo_model
from metrics.detection import map_to_str, map_coco


# import threading

# lock = threading.Lock()

N_WORKERS = 1
Q_SIZE = 10
BATCH_SIZE = 32

async def producer(queue, img_dir, limits: Optional[int] = None):
    batch = []
    if limits is not None:
        img_paths = list(img_dir.iterdir())[:limits]
    else:
        img_paths = img_dir.iterdir()

    for img_path in tqdm(img_paths, mininterval=1):
        img = Image.open(img_path)
        img.load()
        batch.append((img_path.resolve(), img))
        if len(batch) == BATCH_SIZE:
            await queue.put(batch)
            batch = []
    if batch:
        await queue.put(batch)
    await queue.put(None)


async def consumer(queue, model) -> list[Detection]:
    rows_all = []
    while True:
        batch = await queue.get()
        if batch is None:
            break
        loop = asyncio.get_event_loop()
        # with lock:
        #     results = model.detect(img)
        results = await loop.run_in_executor(None, model.detect, [img for _, img in batch])
        rows = []
        for d in results:
            rows.append(
                {
                    "image_name": d.image_name,
                    "class": d.label,
                    "score": float(d.score),
                    "x1": float(d.bbox[0]),
                    "y1": float(d.bbox[1]),
                    "x2": float(d.bbox[2]),
                    "y2": float(d.bbox[3]),
                }
            )
        queue.task_done()
        rows_all.extend(rows)
    return rows_all


async def run_cv(
    models=None,
    input_dir: str = 'data/coco128',
    output_file: str = 'predictions',
    classes: list[str] | None = None,
    limits: Optional[int] = None
):
    img_dir = Path(input_dir)
    queue = asyncio.Queue(maxsize=Q_SIZE)

    tasks = [
        asyncio.create_task(producer(queue, img_dir, limits=limits)),
    ]

    for i in range(N_WORKERS):
        tasks.append(asyncio.create_task(consumer(queue, models[i])))

    results = await asyncio.gather(*tasks)
    rows_all = []
    for r in results[1:]:  # пропускаем producer
        rows_all.extend(r)
    pd.DataFrame(rows_all).to_csv(output_file, index=False)



def evaluate_model_per_img(
    limit: int | None = None,
    model=None,
    input_dir: str = 'data/coco128',
    output_file: str = 'predictions',
    classes: list[str] | None = None
):
    if model is None:
        return pd.DataFrame(columns=['image_name', 'class', 'x1', 'y1', 'x2', 'y2'])

    model.load()
    img_dir = Path(input_dir)
    rows_all = []

    results = model.detect(
        path_to_dir=input_dir
    )
    for result in results:
        for d in result.boxes:
            rows_all.append(
                {
                    "image_name": d.image_name,
                    "class": d.label,
                    "score": float(d.score),
                    "x1": float(d.bbox[0]),
                    "y1": float(d.bbox[1]),
                    "x2": float(d.bbox[2]),
                    "y2": float(d.bbox[3])
                }
            )
        if classes:
            pass#TODO
    pred = pd.DataFrame(data=rows_all, columns=['image_name', 'class', 'score', 'x1', 'y1', 'x2', 'y2'])
    pred.to_csv(output_file, index=False)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/coco/coco2017/train2017")
    parser.add_argument("--ann_file", type=str, default="data/coco/coco2017/annotations/annotations_train2017.csv")
    parser.add_argument("--output_dir", type=str, default="predictions/coco_yolo.csv")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--q_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    Q_SIZE = args.q_size
    N_WORKERS = args.n_workers
    BATCH_SIZE = args.batch_size

    n_models = []
    for idx in range(N_WORKERS):
        model = yolo_model(model_id=args.model, device=f"cuda:{idx}")
        model.load()
        n_models.append(model)

    input_dir = args.input_dir
    ann_file = args.ann_file
    output_dir = args.output_dir

    # pred = evaluate_model_per_img(model=model, input_dir=input_dir, output_file=output_dir)
    # gt = pd.read_csv(
    #     ann_file
    # )

    # classes = sorted(pd.unique(gt["class"]).tolist())

    # processed_images = pred["image_name"].unique()
    # gt_subset = gt[gt["image_name"].isin(processed_images)].copy()

    # print(map_to_str(map_coco(pred=pred, gt=gt_subset, classes=classes)))
    pred = asyncio.run(
        run_cv(models=n_models,
            input_dir=input_dir,
            output_file=output_dir
        ))

> cp /kaggle/input/models/votter/coursework2/pytorch/default/1/ ./coursework
cp: -r not specified; omitting directory '/kaggle/input/models/votter/coursework2/pytorch/default/1/'

> cp -r /kaggle/input/models/votter/coursework2/pytorch/default/1/ ./coursework
> ls -a
./  ../  coursework/  .virtual_documents/

> mkdir data/coco/coco2017
mkdir: cannot create directory ‘data/coco/coco2017’: No such file or directory

> mkdir data/coco
mkdir: cannot create directory ‘data/coco’: No such file or directory

mkdir data/
mkdir data/coco
mkdir data/coco/coco2017
> !ln -s /kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017 data/coco/coco2017/train2017
> ls 
annotations_train2017.csv  coursework/  data/

> mv annotations_train2017.csv data/coco/coco2017/annotations/annotations_train2017.csv
mv: cannot move 'annotations_train2017.csv' to 'data/coco/coco2017/annotations/annotations_train2017.csv': No such file or directory

> mkdir data/coco/coco2017/annotations
> mkdir predictions
> mv annotations_train2017.csv data/coco/coco2017/annotations/annotations_train2017.csv
> !python run_vlm.py
python3: can't open file '/kaggle/working/run_vlm.py': [Errno 2] No such file or directory

> ls -a
./  ../  coursework/  data/  predictions/  .virtual_documents/

> mv data /coursework/data

> mv data ./coursework/data
> mv predictions ./coursework/predictions
> cd coursework
/kaggle/working/coursework
> ls
data/        metrics/  predictions/    pyrightconfig.json  run_cv.py   utils/

__init__.py  model/    pyproject.toml  README.md           run_vlm.py

# whole_body_detection

## dataset:
- from coco: https://cocodataset.org/#home
and coco-whole-body: https://github.com/jin-s13/COCO-WholeBody

- add keypoints from 17(coco) to 35(6 for each hand, 3 for each f00t, total: 18, from coco-whole-body)
- more detail see load_ann.py

## model:
- detection part: using faster rcnn and load checkpoints from pretrained model.
- pose part: using resNet_50 and change input size as 256x256, output size as 64x64. Trained from pretrained model for 17 coco model.
- more detail see train.py

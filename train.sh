# python train.py --img 640 --batch 32 --epochs 100 --weights weights/yolov5s.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand --device 0 --optimizer AdamW

python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64 --device 3 --epochs 100 --name coco --optimizer AdamW --data data/coco.yaml
# python prune.py --percent 0 --weights runs/train/coco_hand_sparsity6/weights/last.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --imgsz 640
python prune.py --percent 0.5 --weights runs/train/coco_sparsity2/weights/last.pt --data data/coco.yaml --cfg models/yolov5s.yaml --imgsz 640

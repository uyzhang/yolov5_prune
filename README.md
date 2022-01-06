### Introduction
Clean code version of [yolov5](https://github.com/ultralytics/yolov5/)(V6) pruning.

The original code comes from : https://github.com/midasklr/yolov5prune.

### Steps:
1. Dataset preparation
    [COCO Hand](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz) Dataset Download.

    Convert dataset to trainable format : [converter](https://github.com/ZJU-lishuang/yolov5-v4/blob/main/data/converter.py).

2. Basic training
    ```shell
    python train.py --img 640 --batch 32 --epochs 100 --weights weights/yolov5s.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand --device 0 --optimizer AdamW
    ```

3. Sparse training
    ```shell
    python train.py --img 640 --batch 32 --epochs 100 --weights runs/train/coco_hand/weights/last.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand_sparsity --optimizer AdamW --bn_sparsity --sparsity_rate 0.0001 --device 3
    ```

4. Pruning
    ```shell
    python prune.py --percent 0.5 --weights runs/train/coco_hand_sparsity6/weights/last.pt --data data/coco_hand.yaml --cfg models/yolov5s.yaml --imgsz 640
    ```

5. Fine-tuning
    ```shell
    python train.py --img 640 --batch 32 --epochs 100 --weights runs/val/exp2/pruned_model.pt  --data data/coco_hand.yaml --cfg models/yolov5s.yaml --name coco_hand_ft --device 0 --optimizer AdamW --ft_pruned_model --hyp hyp.finetune_prune.yaml
    ```
### Experiments
TODO



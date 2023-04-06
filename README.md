### Introduction
Clean code version of [YOLOv5](https://github.com/ultralytics/yolov5/)(V6) pruning.

The original code comes from : https://github.com/midasklr/yolov5prune.

### Steps:
1. Basic training
    - In COCO Dataset
        ```shell
        python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 32 --device 0 --epochs 300 --name coco --optimizer AdamW --data data/coco.yaml
        ```
2. Sparse training
    - In COCO Dataset
        ```shell
        python train.py --batch 32 --epochs 50 --weights weights/yolov5s.pt --data data/coco.yaml --cfg models/yolov5s.yaml --name coco_sparsity --optimizer AdamW --bn_sparsity --sparsity_rate 0.00005 --device 0
        ```

3. Pruning
    - In COCO Dataset
        ```shell
        python prune.py --percent 0.5 --weights runs/train/coco_sparsity13/weights/last.pt --data data/coco.yaml --cfg models/yolov5s.yaml --imgsz 640
        ```

4. Fine-tuning
    - In COCO Dataset
        ```shell
        python train.py --img 640 --batch 32 --epochs 100 --weights runs/val/exp1/pruned_model.pt  --data data/coco.yaml --cfg models/yolov5s.yaml --name coco_ft --device 0 --optimizer AdamW --ft_pruned_model --hyp hyp.finetune_prune.yaml
        ```
### Experiments
- Result of COCO Dataset
    | exp\_name        | model   | optim&epoch | lr     | sparity | mAP@.5  | note                | prune threshold | BN weight distribution                                                           | Weight |
    | ---------------- | ------- | ----------- | ------ | ------- | ------- | ------------------- | --------------- | -------------------------------------------------------------------------------- | ------------ |
    | coco             | yolov5s | adamw 100   | 0.01   | \-      | 0.5402  | \-                  | \-              | \-                                                                               | - |
    | coco2            | yolov5s | adamw 300   | 0.01   | \-      | 0.5534  | \-                  | \-              | \-                                                                               | [last.pt](https://github.com/uyzhang/yolov5_prune/releases/download/ckp/coco_adamw_300.pt)   |
    | coco\_sparsity   | yolov5s | adamw 50    | 0.0032 | 0.0001  | 0.4826  | resume official SGD | 0.54            | ![](https://docimg8.docs.qq.com/image/37lM2bxXOohzeYLQzhsU0g.png?w=1322&h=826/)  | \-           |
    | coco\_sparsity2  | yolov5s | adamw 50    | 0.0032 | 0.00005 | 0.50354 | resume official SGD | 0.48            | ![](https://docimg8.docs.qq.com/image/fsUuusfnXh0QqNIzBsQorA.png?w=1342&h=822/)  | \-           |
    | coco\_sparsity3  | yolov5s | adamw 50    | 0.0032 | 0.0005  | 0.39514 | resume official SGD | 0.576           | ![](https://docimg10.docs.qq.com/image/56lYy7Ig1U9aKtv3JoaVuw.png?w=1330&h=864/) | \-           |
    | coco\_sparsity4  | yolov5s | adamw 50    | 0.0032 | 0.001   | 0.34889 | resume official SGD | 0.576           | ![](https://docimg2.docs.qq.com/image/PoOcEBkq8k5yAHHuLMTX2w.png?w=1292&h=852/)  | \-           |
    | coco\_sparsity5  | yolov5s | adamw 50    | 0.0032 | 0.00001 | 0.52948 | resume official SGD | 0.579           | ![](https://docimg7.docs.qq.com/image/8sQYKDSEny6fE1-aD-i1PA.png?w=1308&h=842/)  | \-           |
    | coco\_sparsity6  | yolov5s | adamw 50    | 0.01   | 0.0005  | 0.51202 | resume coco         | 0.564           | ![](https://docimg2.docs.qq.com/image/mi5sH-NIcOfhCA5UvblkGQ.png?w=1314&h=758/)  | \-           |
    | coco\_sparsity10 | yolov5s | adamw 50    | 0.01   | 0.001   | 0.49504 | resume coco2        | 0.6             | ![](https://docimg10.docs.qq.com/image/IHpHc5QDZlH4qvX8C14-Uw.png?w=1326&h=826/) | \-           |
    | coco\_sparsity11 | yolov5s | adamw 50    | 0.01   | 0.0005  | 0.52609 | resume coco2        | 0.6             | ![](https://docimg8.docs.qq.com/image/txnqJ5L1PjO96e2DvMPuFQ.png?w=1320&h=826/)  | \-           |
    | coco\_sparsity13 | yolov5s | adamw 100   | 0.01   | 0.0005  | 0.533   | resume coco2        | 0.55            | ![](https://docimg2.docs.qq.com/image/Y0eW6Fg3GxQDNT0pUcHqZw.png?w=1314&h=768/)  | [last.pt](https://github.com/uyzhang/yolov5_prune/releases/download/ckp/coco_sparsity13.pt)           |
    | coco\_sparsity14 | yolov5s | adamw 50    | 0.01   | 0.0007  | 0.515   | resume coco2        | 0.61            | ![](https://docimg7.docs.qq.com/image/uI9OFouJavwCSGAK8kk8vg.png?w=1312&h=782/)  | \-           |
    | coco\_sparsity15 | yolov5s | adamw 100   | 0.01   | 0.001   | 0.501   | resume coco2        | 0.54            | ![](https://docimg4.docs.qq.com/image/wyGMs5I4U_8vsXQLgG6LJg.png?w=1304&h=820/)  | \-           |

- The model of pruning coco_sparsity13
    | coco_sparsity13   | mAP@.5 | Params/FLOPs |
    |-------------------|--------|--------------|
    | origin            | 0.537  | 7.2M/16.5G   |
    | after 10% prune   | 0.5327 | 6.2M/15.6G   |
    | after 20% prune   | 0.5327 | 5.4M/14.7G   |
    | after 30% prune   | 0.5324 | 4.4M/13.8G   |
    | after 33% prune   | 0.5281 | 4.2M/13.6G   |
    | after 34% prune   | 0.5243 | 4.18M/13.5G  |
    | after 34.5% prune | 0.5203 | 4.14M/13.5G  |
    | after 35% prune   | 0.2548 | 4.1M/13.4G   |
    | after 38% prune   | 0.2018 | 3.88M/13.0G  |
    | after 40% prune   | 0.1622 | 3.7M/12.7G   |
    | after 42% prune   | 0.1194 | 3.6M/12.4G   |
    | after 45% prune   | 0.0537 | 3.4M/12.0G   |
    | after 50% prune   | 0.0032 | 3.1M/11.4G   |

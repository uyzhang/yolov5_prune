python export.py --weights runs/val/exp8/pruned_model_0.2.pt --include engine --device 0
python export.py --weights runs/val/exp12/pruned_model_0.4.pt --include engine --device 0
python export.py --weights runs/train/coco_hand_sparsity6/weights/last.pt --include engine --device 0
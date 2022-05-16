# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

from operator import mod
from models.yolo import *
from utils.torch_utils import select_device, de_parallel
from utils.general import (check_dataset, check_img_size, check_yaml,
                           colorstr, increment_path, print_args)
from utils.datasets import create_dataloader
from utils.callbacks import Callbacks
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from utils.prune_utils import get_mask_bn, get_prune_threshold, get_bn_list, get_pruned_yaml, prune_model_load_weight
import val

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.no_grad()
def prune(data,
          weights=None,  # model.pt path(s)
          cfg='models/yolov5l.yaml',
          percent=0,
          batch_size=32,  # batch size
          imgsz=640,  # inference size (pixels)
          conf_thres=0.001,  # confidence threshold
          iou_thres=0.6,  # NMS IoU threshold
          task='val',  # train, val, test, speed or study
          device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
          workers=8,  # max dataloader workers (per RANK in DDP mode)
          single_cls=False,  # treat as single-class dataset
          augment=False,  # augmented inference
          verbose=False,  # verbose output
          save_txt=False,  # save results to *.txt
          save_hybrid=False,  # save label+prediction hybrid results to *.txt
          save_conf=False,  # save confidences in --save-txt labels
          save_json=False,  # save a COCO-JSON results file
          project=ROOT / 'runs/val',  # save to project/name
          name='exp',  # save to project/name
          exist_ok=False,  # existing project/name ok, do not increment
          half=True,  # use FP16 half-precision inference
          dnn=False,  # use OpenCV DNN for ONNX inference
          model=None,
          dataloader=None,
          save_dir=Path(''),
          plots=True,
          callbacks=Callbacks(),
          compute_loss=None,
          val_in_prune=True,
          ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        # get model device, PyTorch model
        device, pt, jit, engine = next(
            model.parameters()).device, True, False, False

        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fuse=False)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        data = check_dataset(data)  # check

    # Configure
    model = model.model

    model.eval()

    # prune model start
    model_list, ignore_bn_list = get_bn_list(model)

    # replace origin yaml with pruned yaml
    pruned_yaml = get_pruned_yaml(cfg, model.model[-1].nc)
    # bn weight need to be pruned(masked)
    model, mask_bn = get_mask_bn(model, ignore_bn_list, get_prune_threshold(model_list, percent))
    pruned_model = Model(cfg=pruned_yaml, ch=3, mask_bn=mask_bn).cuda()

    # Compatibility updates
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    pruned_model = prune_model_load_weight(model, pruned_model, mask_bn)
    pruned_model.names = model.names
    # prune model end

    torch.save({'model': deepcopy(de_parallel(pruned_model)).half(), }, save_dir / "pruned_model.pt")
    pruned_model.cuda().eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset

    # Dataloader
    if not training:
        if device.type != 'cpu':
            pruned_model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(pruned_model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5

        task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]
    if val_in_prune:
        results, _, _ = val.run(data,
                                batch_size=batch_size,
                                imgsz=imgsz,
                                model=pruned_model,
                                iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                                single_cls=single_cls,
                                dataloader=dataloader,
                                save_dir=save_dir,
                                save_json=is_coco,
                                verbose=True,
                                plots=True,
                                callbacks=callbacks,
                                compute_loss=compute_loss)
        return results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/voc.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT /
                        'runs/train/exp47/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--cfg', type=str,
                        default='models/yolov5l.yaml', help='model.yaml path')
    parser.add_argument('--percent', type=float,
                        default=0.4, help='prune percentage')
    parser.add_argument('--batch-size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size',
                        type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val',
                        help='train, val, test, speed or study')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true',
                        help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--verbose', action='store_true',
                        help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true',
                        help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true',
                        help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT /
                        'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--val_in_prune', action='store_true')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main():
    opt = parse_opt()
    params = vars(opt)
    params_prune = params.copy()
    params.pop('cfg')
    params.pop('percent')
    if not opt.val_in_prune:
        prune(**params_prune)
        return
    params.pop('val_in_prune')
    results_origin, _, _ = val.run(**params)
    results_prune = prune(**params_prune)
    names = ['P', 'R', 'mAP@.5', 'mAP@.5:.95']
    print("=" * 100)
    for (name, o, p) in zip(names, results_origin, results_prune):
        print('|\t {:<10} | origin:{:<10.4f} | after prune:{:<10.4f} | loss ratio:{:<10.4f}'.format(
            name, o, p, (o - p) / o))
    print("=" * 100)


if __name__ == "__main__":
    main()

import torch
from models.common import Bottleneck
import yaml
from torch import nn


def get_bn_list(model):
    model_list = {}
    ignore_bn_list = []

    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:
                ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
        if isinstance(layer, torch.nn.BatchNorm2d):
            model_list[i] = layer
    model_list = {k: v for k, v in model_list.items()
                  if k not in ignore_bn_list}
    return model_list, ignore_bn_list


def get_prune_threshold(model_list, percent):
    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())

    highest_thre = min(highest_thre)
    # 找到highest_thre对应的下标对应的百分比
    percent_threshold = (sorted_bn == highest_thre).nonzero().squeeze().item() / len(bn_weights)
    print('Suggested Gamma threshold should be less than {}'.format(highest_thre))
    print('The corresponding prune ratio is {}, but you can set higher'.format(percent_threshold))
    thre_index = int(len(sorted_bn) * percent)
    thre_prune = sorted_bn[thre_index]
    print('Gamma value that less than {} are set to zero'.format(thre_prune))
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    return thre_prune


def get_pruned_yaml(cfg, nc):
    def replace_name(origin_name):
        name_list = []
        for i in origin_name:
            for j in range(len(i)):
                if i[j] == 'C3':
                    i[j] = 'C3Pruned'
                if i[j] == 'SPPF':
                    i[j] = 'SPPFPruned'
            name_list.append(i)
        return name_list

    # save pruned model config yaml
    pruned_yaml = {}
    with open(cfg, encoding='ascii', errors='ignore') as f:
        origin_yaml = yaml.safe_load(f)  # model dict

    pruned_yaml["nc"] = nc
    pruned_yaml["depth_multiple"] = origin_yaml["depth_multiple"]
    pruned_yaml["width_multiple"] = origin_yaml["width_multiple"]
    pruned_yaml["anchors"] = origin_yaml["anchors"]
    pruned_yaml["backbone"] = replace_name(origin_yaml["backbone"])
    pruned_yaml["head"] = replace_name(origin_yaml["head"])
    return pruned_yaml


def get_mask_bn(model, ignore_bn_list, thre_prune):
    remain_num = 0
    mask_bn = {}
    for bnname, bnlayer in model.named_modules():
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            mask = obtain_bn_mask(bn_module, thre_prune)
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size()).cuda()
            mask_bn[bnname] = mask
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            bn_module.bias.data.mul_(mask)
            print(f"|\t{bnname:<25}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
            assert int(mask.sum(
            )) > 0, "Current remaining channel must greater than 0!!! please set prune percent to lower thesh, or you can retrain a more sparse model..."
    print("=" * 94)
    return model, mask_bn


def gather_bn_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]
    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        bn_weights[index:(index + size)] = idx.weight.data.abs().clone()
        index += size
    return bn_weights


def gather_conv_weights(module_list):
    size_list = [idx.weight.data.shape[0] for idx in module_list.values()]

    conv_weights = torch.zeros(sum(size_list))
    index = 0
    for i, idx in enumerate(module_list.values()):
        size = size_list[i]
        conv_weights[index:(index + size)] = idx.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).clone()
        index += size
    return conv_weights


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def obtain_conv_mask(conv_module, thre):
    thre = thre.cuda()
    mask = conv_module.weight.data.abs().sum(dim=1).sum(dim=1).sum(dim=1).ge(thre).float()
    return mask

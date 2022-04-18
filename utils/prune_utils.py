import torch
from models.common import Bottleneck
import yaml
from torch import nn
import numpy as np


def get_bn_weights(model, ignore_bn_list):
    module_list = []
    for j, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d) and j not in ignore_bn_list:
            bnw = layer.state_dict()['weight']
            module_list.append(bnw)

    size_list = [idx.data.shape[0] for idx in module_list]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)
                   ] = module_list[idx].data.abs().clone()
        index += size
    return bn_weights


def get_ignore_bn(model):
    ignore_bn_list = []
    for k, m in model.named_modules():
        if isinstance(m, Bottleneck):
            if m.add:
                ignore_bn_list.append(
                    k.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(k + '.cv1.bn')
                ignore_bn_list.append(k + '.cv2.bn')
    return ignore_bn_list


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
    threshold_index = (sorted_bn == highest_thre).nonzero().squeeze()
    if len(threshold_index.shape) > 0:
        threshold_index = threshold_index[0]
    percent_threshold = threshold_index.item() / len(bn_weights)
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


def prune_model_load_weight(model, pruned_model, mask_bn):
    model_state = model.state_dict()
    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()

    assert pruned_model_state.keys() == model_state.keys()
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(), pruned_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(
                        mask_bn[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(
                        np.asarray(mask_bn[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()

                    if len(w.shape) == 3:     # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()

                    pruned_layer.weight.data = w.clone()
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [model_state[i + ".weight"].shape[0]
                               for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(
                            mask_bn[name].shape[0]) if mask_bn[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(
                        mask_bn[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(
                    mask_bn[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(
                np.asarray(mask_bn[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(
                np.asarray(mask_bn[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
    return pruned_model


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

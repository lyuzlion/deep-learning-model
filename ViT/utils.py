import os
import torch
import shutil
import numpy as np
from torchvision import transforms, datasets
import torch
from torch import nn
from model import (vit_base_patch16_224_in21k,
                   vit_base_patch32_224_in21k,
                   vit_large_patch16_224_in21k,
                   vit_large_patch32_224_in21k,
                   vit_huge_patch14_224_in21k)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):
    if args.model == "vit_base_patch16_224":
        model = vit_base_patch16_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_base_patch32_224":
        model = vit_base_patch32_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_large_patch16_224":
        model = vit_large_patch16_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_large_patch32_224":
        model = vit_large_patch32_224_in21k(args.num_classes, has_logits=False)
    elif args.model == "vit_huge_patch14_224":
        model = vit_huge_patch14_224_in21k(args.num_classes, has_logits=False)
    else:
        raise Exception("Can't find any model name call {}".format(args.model))

    return model


def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model


def remove_dir_and_create_dir(dir_name):
    """
    清除原有的文件夹，并且创建对应的文件目录
    Args:
        dir_name: 该文件夹的名字

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


def get_data_loader(data_dir, batch_size, aug=False):
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["train" if aug else "val"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=aug)
    return loader, dataset
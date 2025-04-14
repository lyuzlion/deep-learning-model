import os
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from config import args
from dataloader import get_data_loader
from utils import remove_dir_and_create_dir, create_model, model_parallel, set_seed


device = 'cuda'

weights_dir = args.summary_dir + "/weights"
log_dir = args.summary_dir + "/logs"

remove_dir_and_create_dir(weights_dir)
remove_dir_and_create_dir(log_dir)
writer = SummaryWriter(log_dir)

set_seed(777)

train_loader, train_dataset = get_data_loader(args.dataset_train_dir, args.batch_size, aug=True)
val_loader, val_dataset = get_data_loader(args.dataset_val_dir, args.batch_size)
train_num, val_num = len(train_dataset), len(val_dataset)

model = create_model(args)

if args.weights != "":
    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    weights_dict = torch.load(args.weights, map_location=device)
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))

if args.freeze_layers:
    for name, params in model.named_parameters():
        # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            params.requires_grad_(False)
        else:
            print("training {}".format(name))

model = model_parallel(args, model)
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-5)
lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

best_acc = 0.0

for epoch in range(args.epochs):
    model.train()
    train_acc = 0
    train_loss = []
    train_bar = tqdm(train_loader)
    for images, labels in train_bar:
        train_bar.set_description("epoch {}".format(epoch))
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        prediction = torch.max(logits, dim=1)[1]

        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())
        train_bar.set_postfix(loss="{:.4f}".format(loss.item()))
        train_acc += torch.eq(labels, prediction).sum()

    # validate
    model.eval()
    val_acc = 0
    val_loss = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            prediction = torch.max(logits, dim=1)[1]

            val_loss.append(loss.item())
            val_acc += torch.eq(labels, prediction).sum()


    val_accurate = val_acc / val_num
    train_accurate = train_acc / train_num
    print("=> loss: {:.4f}   acc: {:.4f}   val_loss: {:.4f}   val_acc: {:.4f}".format(np.mean(train_loss), train_accurate, np.mean(val_loss), val_accurate))

    writer.add_scalar("train_loss", np.mean(train_loss), epoch)
    writer.add_scalar("train_acc", train_accurate, epoch)
    writer.add_scalar("val_loss", np.mean(val_loss), epoch)
    writer.add_scalar("val_acc", val_accurate, epoch)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(model.state_dict(), "{}/epoch={}_val_acc={:.4f}.pth".format(weights_dir, epoch, val_accurate))

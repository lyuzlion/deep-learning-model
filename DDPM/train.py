import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
import logging
from torch.utils.tensorboard import SummaryWriter
from model import UNet, Diffusion, EMA
import torch
import torch.nn as nn
import copy
import numpy as np
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


parser = argparse.ArgumentParser()
args = parser.parse_args()

args.epochs = 300
args.batch_size = 16
args.image_size = 64
args.num_classes = 10
args.dataset_path = r"/home/liuzilong/data/liuzilong/animals10"
args.device = "cuda"
args.lr = 3e-4
args.save_path = "/home/liuzilong/data/liuzilong/checkpoints/DDPM/"

setup_logging()
device = args.device
dataloader = get_data(args)
model = UNet(num_classes=args.num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
mse = nn.MSELoss()
diffusion = Diffusion(img_size=args.image_size, device=device)
logger = SummaryWriter("runs")
l = len(dataloader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

for epoch in range(args.epochs):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(dataloader)
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device) # [16, 3, 64, 64]
        labels = labels.to(device) # [16]
        t = diffusion.sample_timesteps(images.shape[0]).to(device) # [16]
        x_t, noise = diffusion.noise_images(images, t) # [16, 3, 64, 64], [16, 3, 64, 64]
        if np.random.random() < 0.1:
            labels = None
        predicted_noise = model(x_t, t, labels)
        loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)

        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    if epoch % 10 == 0:
        labels = torch.arange(args.num_classes).long().to(device)
        sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
        ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
        plot_images(sampled_images)
        save_images(sampled_images, os.path.join("results", f"{epoch}.jpg"))
        save_images(ema_sampled_images, os.path.join("results", f"{epoch}_ema.jpg"))
        torch.save(model.state_dict(), args.save_path + "ckpt.pt")
        torch.save(ema_model.state_dict(), args.save_path + "ema_ckpt.pt")
        torch.save(optimizer.state_dict(), args.save_path + "optim.pt")
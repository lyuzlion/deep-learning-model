
import sys
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm
from model import D3PM, DummyX0Model

if __name__ == "__main__":

    N = 2  # 每个 pixel 的类别
    d3pm = D3PM(DummyX0Model(1, N), 1000, num_classes=N, hybrid_loss_coeff=0.0).cuda()
    print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
    dataset = MNIST(
        root="/home/liuzilong/data/liuzilong/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2), # 将图像大小从28x28扩展到32x32
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=16)
    optim = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=1e-3)
    d3pm.train()
    n_epoch = 400
    device = "cuda"
    global_step = 0

    for i in range(n_epoch):
        progress_bar = tqdm(dataloader)
        loss_ema = None # 指数移动平均（Exponential Moving Average），用于平滑损失值的变化，算一个 trick。
        for x, y in progress_bar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # discritize x to N bins
            x = (x * (N - 1)).round().long().clamp(0, N - 1) # 将输入数据 x 离散化到 [0, N-1] 的整数范围内。要求 x 范围在 01 之间。
            loss, info = d3pm(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 0.1) # 梯度裁剪，防止梯度爆炸。
            with torch.no_grad():
                param_norm = sum([torch.norm(p) for p in d3pm.x0_model.parameters()])
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
            progress_bar.set_description(f"epoch: {i+1}, loss: {loss_ema:.4f}, norm: {norm:.4f}, param_norm: {param_norm:.4f}, vb_loss: {info['vb_loss']:.4f}, ce_loss: {info['ce_loss']:.4f}")
            # ce_loss：交叉熵损失   vb_loss：变分损失
            optim.step()
            global_step += 1

            if global_step % 300 == 1:
                d3pm.eval()
                with torch.no_grad():
                    y = torch.arange(0, 4).cuda() % 10 # target
                    init_noise = torch.randint(0, N, (4, 1, 32, 32)).cuda() # 生成4个随机噪声图像

                    images = d3pm.sample_with_image_sequence(init_noise, y, stride=40)
                    # image sequences to gif
                    gif = []
                    for image in images:
                        x_as_image = make_grid(image.float() / (N - 1), nrow=2)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy() # permute: 调整张量的维度顺序。
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    gif[0].save(
                        f"contents/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"contents/sample_{global_step}_last.png")

                d3pm.train()
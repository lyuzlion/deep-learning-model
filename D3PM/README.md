# D3PM (Discrete Denoising Diffusion Probabilistic Models)
## Diffusion 阶段
逐步加噪，直到产生一张完全噪声图像。
在代码实现中，其实是仅用一步就直接扩散到 $x_t$。
## Reverse 阶段
使用 U-Net，从 $x_t$ 一步去噪到 $x_0$。逐 pixel 算 loss，所以说是离散的。


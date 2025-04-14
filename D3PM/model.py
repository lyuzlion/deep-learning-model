import torch
import torch.nn as nn

blk = lambda input_channel, ouput_channel: nn.Sequential(
    nn.Conv2d(input_channel, ouput_channel, kernel_size=5, padding=2), # p=(k-1)/2，保持输出图像的长宽不变
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    # num_groups（int）：将输入通道分成多个组的数量。每个组都将独立进行归一化，并且具有自己的均值和方差。决定了输入通道的分组数量。
    # num_channels（int）：输入张量的通道数。
    nn.LeakyReLU(),
    nn.Conv2d(ouput_channel, ouput_channel, kernel_size=5, padding=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
    nn.Conv2d(ouput_channel, ouput_channel, kernel_size=5, padding=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
)

# input*N*N 的图像，经过blk之后，变为 output*N*N

blku = lambda input_channel, ouput_channel: nn.Sequential(
    nn.Conv2d(input_channel, ouput_channel, 5, padding=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
    nn.Conv2d(ouput_channel, ouput_channel, 5, padding=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
    nn.Conv2d(ouput_channel, ouput_channel, 5, padding=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
    nn.ConvTranspose2d(ouput_channel, ouput_channel, 2, stride=2),
    nn.GroupNorm(ouput_channel // 8, ouput_channel),
    nn.LeakyReLU(),
) # blku的全称是block up，表示上采样的块。它的作用是将输入的特征图进行上采样，并通过卷积和归一化操作来提取特征。它通常用于生成模型中的解码器部分，以逐步恢复图像的空间分辨率。
'''
卷积
N' = (N − kernel + 2 × padding) / stride + 1
反卷积
N' = (N − 1) × stride + kernel − 2 × padding + outputpadding
'''
class DummyX0Model(nn.Module): # 类似 U-Net，用来将 x_t 变成 x_0
    def __init__(self, n_channel: int, N: int = 16) -> None:
        super(DummyX0Model, self).__init__()
        self.down1 = blk(n_channel, 16)
        self.down2 = blk(16, 32)
        self.down3 = blk(32, 64)
        self.down4 = blk(64, 512)
        self.down5 = blk(512, 512)
        self.up1 = blku(512, 512)
        self.up2 = blku(512 + 512, 64)
        self.up3 = blku(64, 32)
        self.up4 = blku(32, 16)
        self.convlast = blk(16, 16)
        self.final = nn.Conv2d(in_channels=16, out_channels=N * n_channel, kernel_size=1, bias=False)

        self.tr1 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr2 = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.tr3 = nn.TransformerEncoderLayer(d_model=64, nhead=8)

        self.cond_embedding_1 = nn.Embedding(10, 16)
        self.cond_embedding_2 = nn.Embedding(10, 32)
        self.cond_embedding_3 = nn.Embedding(10, 64)
        self.cond_embedding_4 = nn.Embedding(10, 512)
        self.cond_embedding_5 = nn.Embedding(10, 512)
        self.cond_embedding_6 = nn.Embedding(10, 64)

        self.temb_1 = nn.Linear(32, 16)
        self.temb_2 = nn.Linear(32, 32)
        self.temb_3 = nn.Linear(32, 64)
        self.temb_4 = nn.Linear(32, 512)
        self.N = N

    def forward(self, x, t, cond) -> torch.Tensor:
        x = (2 * x.float() / self.N) - 1.0 # [256,1,32,32] 归一化到-1 到 1
        t = t.float().reshape(-1, 1) / 1000 # [256, 1]
        t_features = [torch.sin(t * 3.1415 * 2**i) for i in range(16)] + [
            torch.cos(t * 3.1415 * 2**i) for i in range(16)
        ] # 32 个 256*1 的向量
        tx = torch.cat(t_features, dim=1).to(x.device) # 向右拼接，[256, 32]

        t_emb_1 = self.temb_1(tx).unsqueeze(-1).unsqueeze(-1) # [256, 16, 1, 1]，时间嵌入，不然只有一个模型没法区别不同时刻的图像
        t_emb_2 = self.temb_2(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_3 = self.temb_3(tx).unsqueeze(-1).unsqueeze(-1)
        t_emb_4 = self.temb_4(tx).unsqueeze(-1).unsqueeze(-1)

        cond_emb_1 = self.cond_embedding_1(cond).unsqueeze(-1).unsqueeze(-1) # [256, 16, 1, 1]
        cond_emb_2 = self.cond_embedding_2(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_3 = self.cond_embedding_3(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_4 = self.cond_embedding_4(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_5 = self.cond_embedding_5(cond).unsqueeze(-1).unsqueeze(-1)
        cond_emb_6 = self.cond_embedding_6(cond).unsqueeze(-1).unsqueeze(-1)

        x1 = self.down1(x) + t_emb_1 + cond_emb_1 # 利用广播，虽然形状不同，也能相加 [256, 16, 32, 32]
        x2 = self.down2(nn.functional.avg_pool2d(input=x1, kernel_size=2)) + t_emb_2 + cond_emb_2 # [256, 32, 16, 16]
        x3 = self.down3(nn.functional.avg_pool2d(input=x2, kernel_size=2)) + t_emb_3 + cond_emb_3 # [256, 64, 8, 8]
        x4 = self.down4(nn.functional.avg_pool2d(input=x3, kernel_size=2)) + t_emb_4 + cond_emb_4 # [256, 512, 4, 4]
        x5 = self.down5(nn.functional.avg_pool2d(input=x4, kernel_size=2)) # [256, 512, 2, 2]

        x5 = self.tr1(x5.reshape(x5.shape[0], x5.shape[1], -1).transpose(1, 2)).transpose(1, 2).reshape(x5.shape) # [256, 512, 2, 2]

        y = self.up1(x5) + cond_emb_5 # [256, 512, 4, 4]
        y = self.tr2(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2)).transpose(1, 2).reshape(y.shape)
        y = self.up2(torch.cat([x4, y], dim=1)) + cond_emb_6 # U-Net 的跳跃连接  [256, 64, 8, 8]
        y = self.tr3(y.reshape(y.shape[0], y.shape[1], -1).transpose(1, 2)).transpose(1, 2).reshape(y.shape)
        y = self.up3(y)  # [256, 32, 16, 16]
        y = self.up4(y)  # [256, 16, 32, 32]
        y = self.convlast(y)  # [256, 16, 32, 32]
        y = self.final(y)  # [256, 2, 32, 32]

        # reshape to B, C, H, W, N
        y = y.reshape(y.shape[0], -1, self.N, *x.shape[2:]).transpose(2, -1).contiguous()
        return y # [256, 1, 32, 32, 2]


class D3PM(nn.Module):
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model

        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff

        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999)

        self.eps = 1e-6
        self.num_classses = num_classes
        q_onestep_mats = []
        q_mats = []  # these are cumulative

        for beta in self.beta_t:
            if forward_type == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)

        q_one_step_transposed = q_one_step_mats.transpose(1, 2)  # this will be used for q_posterior_logits

        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.logit_type = "logit"

        # register
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)

        assert self.q_mats.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.q_mats.shape

    def _at(self, a, t, x):
        bs = t.shape[0]
        t = t.reshape((bs, *[1] * (x.dim() - 1)))
        return a[t - 1, x, :] # a[t-1, x, :]。a[0,0,i,j,0/1] 表示位置在(i, j)的像素转移成0/1的概率。

    def q_posterior_logits(self, x_0, x_t, t):

        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps)
        else:
            x_0_logits = x_0.clone()

        assert x_0_logits.shape == x_t.shape + (self.num_classses,), print(f"x_0_logits.shape: {x_0_logits.shape}, x_t.shape: {x_t.shape}")

        fact1 = self._at(self.q_one_step_transposed, t, x_t)

        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        qmats2 = self.q_mats[t - 2].to(dtype=softmaxed.dtype)
        fact2 = torch.einsum("b...c,bcd->b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))

        bc = torch.where(t_broadcast == 1, x_0_logits, out)

        return bc

    def vb(self, dist1, dist2):

        dist1 = dist1.flatten(start_dim=0, end_dim=-2)
        dist2 = dist2.flatten(start_dim=0, end_dim=-2)

        out = torch.softmax(dist1 + self.eps, dim=-1) * (torch.log_softmax(dist1 + self.eps, dim=-1) - torch.log_softmax(dist2 + self.eps, dim=-1))
        return out.sum(dim=-1).mean()

    def q_sample(self, x_0, t, noise):
        # forward process, x_0 is the clean input.
        logits = torch.log(self._at(self.q_mats, t, x_0) + self.eps) # [256,1,32,32,2]
        noise = torch.clip(input=noise, min=self.eps, max=1.0) # [256,1,32,32,2]
        gumbel_noise = -torch.log(-torch.log(noise)) # [256,1,32,32,2]
        return torch.argmax(logits + gumbel_noise, dim=-1) # [256,1,32,32]，argmax的意思是x_0转移成了0还是1
        # 这里的意思是，x_0经过q_mats的转移，变成了x_t。x_t是一个离散的分布，表示每个像素点在不同类别之间的概率分布。


    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """
        Makes forward diffusion x_t from x_0, and tries to guess x_0 value from x_t using x0_model.
        x is one-hot of dim (bs, ...), with int values of 0 to num_classes - 1
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device) # [256]，每个数都是1到n_T-1之间的随机数
        x_t = self.q_sample(x, t, torch.rand((*x.shape, self.num_classses), device=x.device)) # [256, 1, 32, 32]，得到 t 时刻的图像
        # x_t is same shape as x
        assert x_t.shape == x.shape, print(
            f"x_t.shape: {x_t.shape}, x.shape: {x.shape}"
        )
        # we use hybrid loss.

        predicted_x0_logits = self.x0_model(x_t, t, cond) # [256, 1, 32, 32, 2] 这模拟的是反向的过程，得到x_0的每个像素属于0/1的logits。

        # based on this, we first do vb loss.
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t) # [256, 1, 32, 32, 2]
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t) # [256, 1, 32, 32, 2]

        vb_loss = self.vb(true_q_posterior_logits, pred_q_posterior_logits)

        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2) # [262144, 2]
        x = x.flatten(start_dim=0, end_dim=-1) # [262144]

        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        # return self.hybrid_loss_coeff * vb_loss + ce_loss # for visualization
        return self.hybrid_loss_coeff * vb_loss + ce_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise):
        predicted_x0_logits = self.model_predict(x, t, cond)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t)

        noise = torch.clip(noise, self.eps, 1.0)

        not_first_step = (t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))

        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1)
        return sample

    def sample(self, x, cond=None):
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device))
        return x

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        for t in reversed(range(1, self.n_T)):
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x = self.p_sample(x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device))
            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)
        return images
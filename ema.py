import torch
import torch.nn.functional as F


def torch_ema(price_mat, span, device='cuda'):
    price = torch.as_tensor(price_mat, dtype=torch.float32, device=device)
    alpha = 2.0 / (span + 1)

    S, T = price.shape
    x = price.unsqueeze(1)

    # 构造指数权重（倒序，使 conv1d = EMA）
    weights = (1 - alpha) ** torch.arange(span, device=device)
    weights = alpha * weights
    kernel = weights.view(1, 1, -1)

    # 卷积计算 EMA（valid）
    y = F.conv1d(x, kernel)

    pad = torch.full((S, 1, span-1), float('nan'), device=device)
    y_full = torch.cat([pad, y], dim=2)

    return y_full.squeeze(1)

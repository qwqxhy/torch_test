import numpy as np

def moving_average_matrix(price, window):
    price = np.asarray(price)
    T = len(price)

    # 构造 Rolling 窗口矩阵 (T - window + 1, window)
    shape = (T - window + 1, window)
    strides = (price.strides[0], price.strides[0])
    windows = np.lib.stride_tricks.as_strided(price, shape=shape, strides=strides)

    # 矩阵乘法 = 每行求和 / window
    ma = windows.dot(np.ones(window)) / window

    # 前 window-1 个位置没有 MA
    ma = np.concatenate([np.full(window-1, np.nan), ma])
    return ma


import torch
import torch.nn.functional as F

def torch_ma(price_mat, window, device='cuda'):
    """
    price_mat: numpy 或 torch 数组，shape = (S, T)
    window: MA 窗口，如 5/10/20
    """
    # 转 torch，放 GPU
    price = torch.as_tensor(price_mat, dtype=torch.float32, device=device)
    S, T = price.shape          # S=股票数, T=时间长度

    # conv1d 需要 3D: (batch, channel, length)
    x = price.unsqueeze(1)      # shape = (S, 1, T)

    # 卷积核：shape = (out_channels=1, in_channels=1, kernel_size=window)
    kernel = torch.ones(1, 1, window, device=device) / window

    # Conv1d = 有效卷积 (valid)，输出长度 = T - window + 1
    y = F.conv1d(x, kernel)     # shape = (S, 1, T-window+1)

    # 前面 window-1 个位置补 NaN 用来对齐
    pad = torch.full((S, 1, window-1), float('nan'), device=device)
    y_full = torch.cat([pad, y], dim=2)  # shape = (S, 1, T)

    return y_full.squeeze(1)             # shape = (S, T)
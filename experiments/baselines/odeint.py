import torch
import numpy as np


def custom_rk4(func, y1, t, step_size=0.01):
    x1, v1 = y1
    xs = [x1]
    vs = [v1]
    if (t[1] - t[0]).item() * step_size < 0:
        step_size = -step_size
    t_list = torch.arange(t[0], t[1], step_size)
    for t in t_list:
        x = xs[-1]
        v = vs[-1]
        k1 = func(t, x, v)
        k2 = func(t + step_size / 2, x + step_size / 2 * k1, k1)
        k3 = func(t + step_size / 2, x + step_size / 2 * k2, k2)
        k4 = func(t + step_size, x + step_size * k3, k3)
        v = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += step_size * v
        xs.append(x)
        vs.append(v)
    return xs

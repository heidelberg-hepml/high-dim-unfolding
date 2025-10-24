import torch
from collections.abc import Callable


def custom_rk4(
    func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    y1: tuple[torch.Tensor, torch.Tensor],
    t: torch.Tensor,
    step_size: float,
) -> list[torch.Tensor]:
    """
    Runge-Kutta 4th order ODE solver with explicit velocity input for self conditioning.
    Parameters
    ----------
        func: function defining the ODE (takes in time, position, velocity, return velocity)
        y1: tuple of (initial position, initial velocity)
        t: torch.Tensor of (start time, end time).
        step_size: integration step size.
    Returns
    -------
        List of torch.Tensor positions at each integration step.
    """
    x1, v1 = y1
    xs = [x1]
    vs = [v1]
    if (t[1] - t[0]).item() * step_size < 0:
        step_size = -step_size
    t_list = torch.arange(
        start=t[0].item(),
        end=t[1].item(),
        step=step_size,
        device=t.device,
        dtype=t.dtype,
    )
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

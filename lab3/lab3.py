import numpy as np
import torch
import torch.nn.functional as F


def load_pgm(path):
    """Read a binary PGM image file.

    path     path to a P5 PGM file
    returns  (H, W) float64 tensor with values in [0, 1]
    """
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        width, height = map(int, f.readline().split())
        assert int(f.readline()) == 255
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    return torch.tensor(raw / 255.0, dtype=torch.float64)


def load_kernel(path):
    """Read a blur kernel from a text file.

    path     path to a kernel file, one row of the kernel per line
    returns  (Ka, Kb) float64 tensor, nonnegative and summing to 1
    """
    return torch.tensor(np.loadtxt(path), dtype=torch.float64)


def forward(x, a):
    """Apply the blur operator A of equation (2).

    x        (H, W) float64 tensor, the image
    a        (Ka, Kb) float64 tensor, the blur kernel, odd sized in both axes
    returns  (H, W) float64 tensor, A x
    """
    ka, kb = a.shape
    pad = (kb // 2, kb // 2, ka // 2, ka // 2)
    x_pad = F.pad(x[None, None], pad, mode="circular")
    kernel = torch.flip(a, dims=(0, 1))[None, None]
    return F.conv2d(x_pad, kernel)[0, 0]


def adjoint_autograd(v, a):
    """Apply A^t, obtained from forward() by automatic differentiation.

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    x = torch.zeros_like(v, requires_grad=True)
    return torch.autograd.grad((forward(x, a) * v).sum(), x)[0]


def adjoint(v, a):
    """Apply the transpose operator A^t of equation (4).

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    ka, kb = a.shape
    pad = (kb // 2, kb // 2, ka // 2, ka // 2)
    v_pad = F.pad(v[None, None], pad, mode="circular")
    kernel = a[None, None]
    return F.conv2d(v_pad, kernel)[0, 0]


_G = torch.tensor([[1 / 12, 1 / 6, 1 / 12],
                    [1 / 6,  0.0,  1 / 6],
                    [1 / 12, 1 / 6, 1 / 12]], dtype=torch.float64)


def prior_grad(x, sigma_x):
    """Return B x, the gradient of prior_cost, from equation (7).

    x        (H, W) float64 tensor
    sigma_x  float
    returns  (H, W) float64 tensor
    """
    g = _G[None, None]
    neighbor_sum = F.conv2d(x[None, None], g, padding=1)[0, 0]
    c = F.conv2d(torch.ones_like(x)[None, None], g, padding=1)[0, 0]
    return (c * x - neighbor_sum) / sigma_x ** 2


def prior_cost(x, sigma_x):
    """Return the prior term (1/2) x^t B x of equation (7).

    x        (H, W) float64 tensor
    sigma_x  float
    returns  float64 scalar tensor
    """
    return 0.5 * (x * prior_grad(x, sigma_x)).sum()


def cost(x, y, a, sigma_w, sigma_x):
    """Return the MAP cost c(x) of equation (8).

    x, y     (H, W) float64 tensors
    a        (Ka, Kb) float64 tensor, the blur kernel
    returns  float64 scalar tensor
    """
    residual = y - forward(x, a)
    data_term = (residual ** 2).sum() / (2 * sigma_w ** 2)
    return data_term + prior_cost(x, sigma_x)


def cost_grad(x, y, a, sigma_w, sigma_x):
    """Return the gradient of the MAP cost, equation (9).

    returns  (H, W) float64 tensor
    """
    return -adjoint(y - forward(x, a), a) / sigma_w ** 2 + prior_grad(x, sigma_x)


def gradient_descent(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Minimize the MAP cost by gradient descent, equation (10).

    Starts at x = y.  Uses the step size of equation (11) when alpha
    is None.

    returns  (x_hat, cost_history), an (H, W) tensor and a list of
             length num_iters + 1 whose first entry is c(y)
    """
    if alpha is None:
        alpha = 1 / (1 / sigma_w ** 2 + 2 / sigma_x ** 2)

    x = y.clone()
    cost_history = [cost(x, y, a, sigma_w, sigma_x).item()]
    with torch.no_grad():
        for _ in range(num_iters):
            x = x - alpha * cost_grad(x, y, a, sigma_w, sigma_x)
            cost_history.append(cost(x, y, a, sigma_w, sigma_x).item())
    return x, cost_history

import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from lab3 import load_pgm, prior_cost, prior_grad, _G

x = load_pgm("kodim23.pgm")

# Task 4a: noncausal prediction error e_s = x_s - sum_{r in dS} g_{s-r} x_r
e = x - F.conv2d(x[None, None], _G[None, None], padding=1)[0, 0]

interior_std = e[1:-1, 1:-1].std()
print(f"prediction error std (interior): {interior_std.item():.6f}")

os.makedirs("results", exist_ok=True)
plt.imsave("results/step4_prediction_error.png",
           (e + 0.5).clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
print("saved figure to results/step4_prediction_error.png")

# Sanity check: a constant image should have ~zero prior cost and gradient
const_img = torch.full((20, 20), 0.37, dtype=torch.float64)
sigma_x = 0.1
const_cost = prior_cost(const_img, sigma_x)
const_grad_max = prior_grad(const_img, sigma_x).abs().max()
print(f"constant-image prior cost:      {const_cost.item():.3e}")
print(f"constant-image max |prior_grad|: {const_grad_max.item():.3e}")

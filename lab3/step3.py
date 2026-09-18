import os

import torch
import matplotlib.pyplot as plt

from lab3 import load_pgm, load_kernel, forward, adjoint, adjoint_autograd

x = load_pgm("kodim23.pgm")
a = load_kernel("levin09_kernels/levin09_kernel1.txt")

print(f"kernel shape: {tuple(a.shape)}")
print(f"kernel sum:   {a.sum().item():.6f}")

# Task 3b: three checks, kernel 1, seed 0
torch.manual_seed(0)
u = torch.randn(512, 768, dtype=torch.float64)
v = torch.randn(512, 768, dtype=torch.float64)

check1 = (adjoint(v, a) - adjoint_autograd(v, a)).abs().max()

lhs = (forward(u, a) * v).sum()
rhs = (u * adjoint(v, a)).sum()
check2 = (lhs - rhs).abs() / lhs.abs()

check3 = (forward(u, a) - adjoint(u, a)).abs().max()

print(f"check 1 (adjoint vs adjoint_autograd, max abs diff): {check1.item():.3e}")
print(f"check 2 (transpose property, relative diff):         {check2.item():.3e}")
print(f"check 3 (forward vs adjoint, max abs diff):           {check3.item():.3e}")

# Task 3c: memory to represent A
H, W = x.shape
N = H * W
nnz_per_row = int((a != 0).sum().item())

dense_bytes = N * N * 8
sparse_bytes = N * nnz_per_row * (8 + 4)
forward_bytes = a.numel() * 8

print(f"N = H * W = {H} * {W} = {N}")
print(f"dense matrix memory:   N^2 * 8 bytes        = {N}^2 * 8 = {dense_bytes:,} bytes "
      f"({dense_bytes / 1e12:.3f} TB)")
print(f"sparse matrix memory:  N * nnz_per_row * 12 bytes = {N} * {nnz_per_row} * 12 = "
      f"{sparse_bytes:,} bytes ({sparse_bytes / 1e6:.3f} MB)")
print(f"forward-function memory: Ka * Kb * 8 bytes  = {a.shape[0]} * {a.shape[1]} * 8 = "
      f"{forward_bytes:,} bytes")

# Task 3a: four-image figure
Ax = forward(x, a)
Atx = adjoint(x, a)
diff = Ax - Atx + 0.5

os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
titles = ["x", "Ax", "A^t x", "Ax - A^t x (+0.5)"]
images = [x, Ax, Atx, diff]
for ax_, img, title in zip(axes, images, titles):
    ax_.imshow(img.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    ax_.set_title(title)
    ax_.axis("off")
fig.tight_layout()
fig.savefig("results/step3_four_images.png", dpi=150)
print("saved figure to results/step3_four_images.png")

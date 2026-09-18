import torch

from lab3 import load_kernel, prior_grad, cost, cost_grad

a = load_kernel("levin09_kernels/levin09_kernel1.txt")

seed = 0
sigma_x = 0.05
sigma_w = 0.05

print(f"seed = {seed}, kernel = 1, dtype = float64")
print(f"sigma_x = {sigma_x}, sigma_w = {sigma_w}")

torch.manual_seed(seed)

# Task 6a, check 4: symmetry of B, u^t (B v) == v^t (B u)
u = torch.randn(32, 32, dtype=torch.float64)
v = torch.randn(32, 32, dtype=torch.float64)

lhs = (u * prior_grad(v, sigma_x)).sum()
rhs = (v * prior_grad(u, sigma_x)).sum()
check4 = (lhs - rhs).abs()
check4_scaled = check4 / lhs.abs()

print(f"check 4 (symmetry of B, |u^t B v - v^t B u|):        {check4.item():.3e}")
print(f"check 4, scale-aware (relative to |u^t B v|):        {check4_scaled.item():.3e}")

# Task 6a, check 5: cost_grad vs torch.autograd gradient of cost
z = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
y_small = torch.randn(32, 32, dtype=torch.float64)

auto = torch.autograd.grad(cost(z, y_small, a, sigma_w, sigma_x), z)[0]
manual = cost_grad(z.detach(), y_small, a, sigma_w, sigma_x)
check5 = (auto - manual).abs().max()
check5_scaled = check5 / auto.abs().max()

print(f"check 5 (cost_grad vs autograd, max abs diff):       {check5.item():.3e}")
print(f"check 5, scale-aware (relative to max |autograd|):   {check5_scaled.item():.3e}")

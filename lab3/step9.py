import math
import os

import torch
import matplotlib.pyplot as plt

from lab3 import load_pgm, cost, cost_grad

torch.manual_seed(0)
device = torch.device("cpu")

seed = 0
sigma_w = 0.05
sigma_x = 0.05
num_iters = 500
step_factor = 4.0

os.makedirs("results", exist_ok=True)

x = load_pgm("kodim23.pgm").to(device)
a_identity = torch.ones((1, 1), dtype=torch.float64, device=device)

torch.manual_seed(seed)
w = torch.randn(x.shape, dtype=torch.float64, device=device) * sigma_w
y = x + w  # not clipped: clipping would break the forward model of (1)

alpha_prescribed = 1 / (1 / sigma_w ** 2 + 2 / sigma_x ** 2)
alpha = step_factor * alpha_prescribed

# ---- Gradient descent with the oversized step, stopping at the first
# nonfinite cost so the log plot and the reported final cost stay meaningful.
x_k = y.clone()
cost_history = [cost(x_k, y, a_identity, sigma_w, sigma_x).item()]
overflow_iter = None
with torch.no_grad():
    for k_iter in range(1, num_iters + 1):
        x_k = x_k - alpha * cost_grad(x_k, y, a_identity, sigma_w, sigma_x)
        c = cost(x_k, y, a_identity, sigma_w, sigma_x).item()
        cost_history.append(c)
        if not math.isfinite(c):
            overflow_iter = k_iter
            break
saw_nonfinite = overflow_iter is not None

initial_cost = cost_history[0]
finite_history = [c for c in cost_history if math.isfinite(c)]
final_finite_cost = finite_history[-1]

# ---- Cost vs. iteration, log vertical axis ----
# Plotted as log10(cost) on a linear axis rather than via ax.semilogy: the
# cost spans ~300 orders of magnitude before overflowing, and matplotlib's
# log-scale tick locator itself overflows at that range.
k = torch.arange(len(finite_history))
log_cost = torch.log10(torch.tensor(finite_history, dtype=torch.float64))
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(k.numpy(), log_cost.numpy())
ax.set_xlabel("k")
ax.set_ylabel("log10 c(x^(k))")
ax.set_title(f"Cost vs. iteration, alpha = {step_factor}x prescribed (sigma_x={sigma_x})")
fig.tight_layout()
fig.savefig("results/step9_cost.png", dpi=150)
print("saved figure to results/step9_cost.png")

print("\nfirst 10 iteration costs:")
for k_iter, c in enumerate(cost_history[:10]):
    print(f"  k={k_iter}: c = {c:.6e}")

print(f"\nsigma_w = {sigma_w}, sigma_x = {sigma_x}")
print(f"prescribed alpha (11) = {alpha_prescribed:.6e}")
print(f"step size used = {step_factor}x prescribed = {alpha:.6e}")
print(f"initial cost c(y) = {initial_cost:.6f}")
print(f"final finite cost = {final_finite_cost:.6f}")
if saw_nonfinite:
    print(f"overflow (nonfinite cost) occurred at iteration k = {overflow_iter}")
else:
    print("overflow (nonfinite cost) occurred = never (all 500 iterations finite)")

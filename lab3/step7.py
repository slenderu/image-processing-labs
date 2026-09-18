import os
import time

import torch
import matplotlib.pyplot as plt

from lab3 import load_pgm, gradient_descent, cost

torch.manual_seed(0)
device = torch.device("cpu")

seed = 0
sigma_w = 0.05
sigma_x_list = [0.01, 0.02, 0.05, 0.1, 0.5]
num_iters = 500
# Round-off tolerance for the monotonicity check, scaled to the size of
# c(y) so it does not depend on the absolute units of the cost.
mono_tol_rel = 1e-9

os.makedirs("results", exist_ok=True)

x = load_pgm("kodim23.pgm").to(device)
a_identity = torch.ones((1, 1), dtype=torch.float64, device=device)

print(f"seed = {seed}, device = {device}, dtype = {x.dtype}")
print(f"sigma_w = {sigma_w}, num_iters = {num_iters}")

torch.manual_seed(seed)
w = torch.randn(x.shape, dtype=torch.float64, device=device) * sigma_w
y = x + w  # not clipped: clipping would break the forward model of (1)

y_rmse = torch.sqrt(((y - x) ** 2).mean()).item()

results = {}
for sigma_x in sigma_x_list:
    alpha = 1 / (1 / sigma_w ** 2 + 2 / sigma_x ** 2)

    t0 = time.perf_counter()
    x_hat, cost_history = gradient_descent(y, a_identity, sigma_w, sigma_x, num_iters)
    elapsed = time.perf_counter() - t0

    cost_history_t = torch.tensor(cost_history, dtype=torch.float64)
    diffs = cost_history_t[1:] - cost_history_t[:-1]
    tol = mono_tol_rel * cost_history_t[0].abs()
    violations = (diffs > tol).nonzero().flatten()
    monotonic = violations.numel() == 0
    worst = diffs.max().item()

    rmse = torch.sqrt(((x_hat - x) ** 2).mean()).item()

    results[sigma_x] = dict(
        alpha=alpha,
        x_hat=x_hat,
        cost_history=cost_history,
        monotonic=monotonic,
        num_violations=int(violations.numel()),
        worst_increase=worst,
        rmse=rmse,
        elapsed=elapsed,
    )

# ---- Images: x, y, and the five restored images, same [0, 1] scale ----
panels = [("x", x), ("y", y)] + [
    (f"x_hat (sigma_x={sx})", results[sx]["x_hat"]) for sx in sigma_x_list
]
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
for ax_, (title, img) in zip(axes, panels):
    ax_.imshow(img.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    ax_.set_title(title, fontsize=9)
    ax_.axis("off")
fig.tight_layout()
fig.savefig("results/step7_images.png", dpi=150)
print("saved figure to results/step7_images.png")

# ---- Convergence figure, sigma_x = 0.05 only ----
sx_conv = 0.05
ch = torch.tensor(results[sx_conv]["cost_history"], dtype=torch.float64)
k = torch.arange(len(ch))
# Floor the gap at the round-off level of c(y) so the log plot does not
# dive to the smallest representable float64 once convergence is reached.
floor = ch[0].abs() * torch.finfo(torch.float64).eps
gap = (ch - ch[-1]).clamp(min=floor)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(k.numpy(), ch.numpy())
ax1.set_xlabel("k")
ax1.set_ylabel("c(x^(k))")
ax1.set_title(f"Cost vs. iteration (sigma_x={sx_conv})")

ax2.semilogy(k.numpy(), gap.numpy())
ax2.set_xlabel("k")
ax2.set_ylabel("c(x^(k)) - c(x^(500))")
ax2.set_title(f"Convergence gap (sigma_x={sx_conv})")

fig.tight_layout()
fig.savefig("results/step7_convergence.png", dpi=150)
print("saved figure to results/step7_convergence.png")

# ---- Monotonicity check ----
print("\nMonotonicity check (cost must decrease every iteration, "
      f"round-off tolerance = {mono_tol_rel:.0e} * |c(y)|):")
for sx in sigma_x_list:
    r = results[sx]
    status = "OK" if r["monotonic"] else "FAILED"
    print(f"  sigma_x = {sx:<5} {status}  "
          f"violations = {r['num_violations']:3d} / {num_iters}  "
          f"worst increase = {r['worst_increase']:.3e}")

# ---- Error table ----
print("\nError table:")
print(f"{'row':<14}{'alpha':>14}{'final cost':>16}{'RMSE':>12}")
print(f"{'y (noisy)':<14}{'':>14}{'':>16}{y_rmse:>12.6f}")
for sx in sigma_x_list:
    r = results[sx]
    print(f"{'sigma_x=' + str(sx):<14}{r['alpha']:>14.6e}"
          f"{r['cost_history'][-1]:>16.6f}{r['rmse']:>12.6f}")

with open("results/step7_rmse_table.csv", "w") as f:
    f.write("row,alpha,final_cost,rmse\n")
    f.write(f"y (noisy),,,{y_rmse:.6f}\n")
    for sx in sigma_x_list:
        r = results[sx]
        f.write(f"sigma_x={sx},{r['alpha']:.6e},{r['cost_history'][-1]:.6f},{r['rmse']:.6f}\n")
print("saved table to results/step7_rmse_table.csv")

# ---- Timing ----
print("\nTiming (wall clock, one 500-iteration gradient_descent run):")
for sx in sigma_x_list:
    print(f"  sigma_x = {sx:<5} {results[sx]['elapsed']:.3f} s")

# ---- Save cost histories for Step 8 ----
torch.save(
    {
        "sigma_w": sigma_w,
        "sigma_x_list": sigma_x_list,
        "cost_histories": {sx: results[sx]["cost_history"] for sx in sigma_x_list},
    },
    "results/step7_cost_histories.pt",
)
print("saved cost histories to results/step7_cost_histories.pt")

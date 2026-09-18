import os
import time

import torch
import matplotlib.pyplot as plt

from lab3 import load_pgm, load_kernel, forward, gradient_descent, cost

torch.manual_seed(0)
device = torch.device("cpu")

seed = 0
sigma_w = 0.02
sigma_x_list = [0.01, 0.05, 0.2]
num_iters = 2000
conv_tol = 1e-3

os.makedirs("results", exist_ok=True)

x = load_pgm("kodim23.pgm").to(device)
a = load_kernel("levin09_kernels/levin09_kernel1.txt").to(device)

print(f"seed = {seed}, device = {device}, dtype = {x.dtype}, kernel = 1")
print(f"sigma_w = {sigma_w}, num_iters = {num_iters}")

# Same blurred noisy image for every sigma_x run.
torch.manual_seed(seed)
w = torch.randn(x.shape, dtype=torch.float64, device=device) * sigma_w
y = forward(x, a) + w  # not clipped: clipping would break the forward model of (1)

y_rmse = torch.sqrt(((y - x) ** 2).mean()).item()


def convergence_iteration(cost_history, num_iters_, tol=conv_tol):
    """Smallest K with c(x^K) - c(x^N) < tol * (c(x^0) - c(x^N))."""
    ch = torch.tensor(cost_history, dtype=torch.float64)
    final = ch[num_iters_]
    gap = ch - final
    target = tol * (ch[0] - final)
    hit = (gap < target).nonzero().flatten()
    return int(hit[0].item()) if hit.numel() else None


results = {}
for sigma_x in sigma_x_list:
    alpha = 1 / (1 / sigma_w ** 2 + 2 / sigma_x ** 2)

    t0 = time.perf_counter()
    x_hat, cost_history = gradient_descent(y, a, sigma_w, sigma_x, num_iters)
    elapsed = time.perf_counter() - t0

    rmse = torch.sqrt(((x_hat - x) ** 2).mean()).item()
    k_conv = convergence_iteration(cost_history, num_iters)

    results[sigma_x] = dict(
        alpha=alpha,
        x_hat=x_hat,
        cost_history=cost_history,
        rmse=rmse,
        elapsed=elapsed,
        k_conv=k_conv,
    )

# ---- Images: y and the three restored images, same [0, 1] scale ----
panels = [("y", y)] + [
    (f"x_hat (sigma_x={sx})", results[sx]["x_hat"]) for sx in sigma_x_list
]
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for ax_, (title, img) in zip(axes.flat, panels):
    ax_.imshow(img.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
    ax_.set_title(title, fontsize=9)
    ax_.axis("off")
fig.tight_layout()
fig.savefig("results/step8_images.png", dpi=150)
print("saved figure to results/step8_images.png")

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
ax2.set_ylabel(f"c(x^(k)) - c(x^({num_iters}))")
ax2.set_title(f"Convergence gap (sigma_x={sx_conv})")

fig.tight_layout()
fig.savefig("results/step8_convergence.png", dpi=150)
print("saved figure to results/step8_convergence.png")

# ---- Error table ----
print("\nError table:")
print(f"{'row':<14}{'alpha':>14}{'final cost':>16}{'RMSE':>12}")
print(f"{'y (blurred+noisy)':<14}{'':>14}{'':>16}{y_rmse:>12.6f}")
for sx in sigma_x_list:
    r = results[sx]
    print(f"{'sigma_x=' + str(sx):<14}{r['alpha']:>14.6e}"
          f"{r['cost_history'][-1]:>16.6f}{r['rmse']:>12.6f}")

with open("results/step8_rmse_table.csv", "w") as f:
    f.write("row,alpha,final_cost,rmse\n")
    f.write(f"y (blurred+noisy),,,{y_rmse:.6f}\n")
    for sx in sigma_x_list:
        r = results[sx]
        f.write(f"sigma_x={sx},{r['alpha']:.6e},{r['cost_history'][-1]:.6f},{r['rmse']:.6f}\n")
print("saved table to results/step8_rmse_table.csv")

# ---- Timing ----
print(f"\nTiming (wall clock, one {num_iters}-iteration gradient_descent run):")
for sx in sigma_x_list:
    print(f"  sigma_x = {sx:<5} {results[sx]['elapsed']:.3f} s")

# ---- Iteration count, sigma_x = 0.05, compared against Step 7 ----
k8 = results[sx_conv]["k_conv"]
print(f"\nIteration count (deconvolution, sigma_x={sx_conv}): "
      f"smallest K with c(x^K) - c(x^{num_iters}) < {conv_tol:.0e} * "
      f"(c(x^0) - c(x^{num_iters})) is K = {k8}")

step7_path = "results/step7_cost_histories.pt"
if os.path.exists(step7_path):
    step7_data = torch.load(step7_path)
    step7_ch = step7_data["cost_histories"][sx_conv]
    step7_num_iters = len(step7_ch) - 1
    k7 = convergence_iteration(step7_ch, step7_num_iters)
    print(f"Iteration count (Step 7 denoising, sigma_x={sx_conv}): "
          f"smallest K with c(x^K) - c(x^{step7_num_iters}) < {conv_tol:.0e} * "
          f"(c(x^0) - c(x^{step7_num_iters})) is K = {k7}")
    print(f"Deconvolution needed {k8 / k7:.1f}x as many iterations as denoising "
          f"to reach the same relative convergence gap (K = {k8} vs. K = {k7}).")

    with open("results/step8_iteration_counts.csv", "w") as f:
        f.write("experiment,sigma_x,num_iters,K\n")
        f.write(f"step8_deconvolution,{sx_conv},{num_iters},{k8}\n")
        f.write(f"step7_denoising,{sx_conv},{step7_num_iters},{k7}\n")
    print("saved iteration counts to results/step8_iteration_counts.csv")
else:
    print(f"Step 7 cost histories not found at {step7_path}; "
          "run step7.py first to enable the comparison.")

    with open("results/step8_iteration_counts.csv", "w") as f:
        f.write("experiment,sigma_x,num_iters,K\n")
        f.write(f"step8_deconvolution,{sx_conv},{num_iters},{k8}\n")
    print("saved iteration counts to results/step8_iteration_counts.csv")

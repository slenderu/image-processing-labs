import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


def step3():
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


def step4():
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


def step6():
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


def step7():
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


def convergence_iteration(cost_history, num_iters_, tol=1e-3):
    """Smallest K with c(x^K) - c(x^N) < tol * (c(x^0) - c(x^N))."""
    ch = torch.tensor(cost_history, dtype=torch.float64)
    final = ch[num_iters_]
    gap = ch - final
    target = tol * (ch[0] - final)
    hit = (gap < target).nonzero().flatten()
    return int(hit[0].item()) if hit.numel() else None


def step8():
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

    results = {}
    for sigma_x in sigma_x_list:
        alpha = 1 / (1 / sigma_w ** 2 + 2 / sigma_x ** 2)

        t0 = time.perf_counter()
        x_hat, cost_history = gradient_descent(y, a, sigma_w, sigma_x, num_iters)
        elapsed = time.perf_counter() - t0

        rmse = torch.sqrt(((x_hat - x) ** 2).mean()).item()
        k_conv = convergence_iteration(cost_history, num_iters, tol=conv_tol)

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
        k7 = convergence_iteration(step7_ch, step7_num_iters, tol=conv_tol)
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


def step9():
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


if __name__ == "__main__":
    step3()
    step4()
    step6()
    step7()
    step8()
    step9()

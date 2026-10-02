"""Lab 4 experiments: checks, D1/D3 shape plots, baseline monotonicity (D5),
runs R1-R6 (D6-D10).  Completed runs are cached in results/runs/ and are
not repeated on a rerun.
"""
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only; avoids a CUDA driver warning
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lab3 import forward, adjoint, load_pgm, load_kernel
from lab4 import (B_WEIGHTS, EPS, rho, rho_prime, neighbor_diffs,
                  surrogate_weights, true_gradient,
                  optimal_step_size, reconstruct)

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")

SEED = 0
SIGMA_W = 0.02
P, Q_EXP = 2.0, 1.2
NUM_ITERS = 50
OMEGA = 1.0

RUNS = {
    "R1": dict(prior="Gaussian MRF", T=100.0, sigma_x=0.036),
    "R2": dict(prior="QGGMRF", T=1.0, sigma_x=0.020),
    "R3": dict(prior="QGGMRF", T=1.0, sigma_x=0.010),
    "R4": dict(prior="QGGMRF", T=1.0, sigma_x=0.040),
    "R5": dict(prior="Gaussian MRF", T=100.0, sigma_x=0.018),
    "R6": dict(prior="Gaussian MRF", T=100.0, sigma_x=0.072),
}

# 128 x 128 crop around the right parrot's beak: rows CROP_R0..CROP_R0+127,
# cols CROP_C0..CROP_C0+127 (0-based, row = vertical index).
CROP_R0, CROP_C0, CROP = 168, 384, 128
PROFILE_ROW = 230  # full-image row index, inside the crop
EDGE_COLS = (398, 420)  # columns spanning the beak's left edge (gray background -> white beak)
FLAT_BOX = (slice(250, 290), slice(385, 400))  # flat gray background patch inside the crop

RES = "results"
RUN_DIR = os.path.join(RES, "runs")

# Reference categorical palette (dataviz skill), slots in fixed order.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "axes.edgecolor": C_MUTED, "axes.labelcolor": C_INK,
    "xtick.color": C_MUTED, "ytick.color": C_MUTED,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10,
})


class Tee:
    def __init__(self, path):
        self.file = open(path, "a")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def rmse(u, v):
    return torch.sqrt(((u - v) ** 2).mean()).item()


def crop(img):
    return img[CROP_R0:CROP_R0 + CROP, CROP_C0:CROP_C0 + CROP]


def gaussian_rho(delta, sigma_x):
    return delta ** 2 / (2 * sigma_x ** 2)


def eq16_gradient(x, y, a, sigma_x, T):
    diffs, valid = neighbor_diffs(x)
    b = torch.tensor(B_WEIGHTS)[:, None, None]
    infl = torch.where(valid, rho_prime(diffs, sigma_x, P, Q_EXP, T), torch.zeros_like(diffs))
    return -adjoint(y - forward(x, a), a) / SIGMA_W ** 2 + (b * infl).sum(0)


def surrogate_cost(x, x_prime, y, a, sigma_x, T):
    """Q(x; x') of equation (13), used only to check g^t H g."""
    b_tilde = surrogate_weights(x_prime, sigma_x, P, Q_EXP, T)
    diffs, _ = neighbor_diffs(x)
    return (((y - forward(x, a)) ** 2).sum() / (2 * SIGMA_W ** 2)
            + 0.5 * (b_tilde * diffs ** 2).sum())


# --------------------------------------------------------------------------
def run_checks(x_true, y, a):
    print("\n=== Correctness checks ===")
    out = {}

    # 1. rho_prime (4) against autograd of rho (3), on a grid that includes 0.
    d = torch.linspace(-0.3, 0.3, 6001).requires_grad_(True)
    worst = 0.0
    for sx, T in [(0.2, 1.0), (0.02, 1.0), (0.01, 1.0), (0.036, 100.0)]:
        auto = torch.autograd.grad(rho(d, sx, P, Q_EXP, T).sum(), d)[0]
        ana = rho_prime(d.detach(), sx, P, Q_EXP, T)
        err = ((auto - ana).abs().max() / ana.abs().max()).item()
        worst = max(worst, err)
        print(f"  rho_prime vs autograd(rho), sigma_x={sx}, T={T}: "
              f"max rel err = {err:.3e}")
    out["rho_prime_vs_autograd_max_rel"] = worst

    # 2. Values at Delta = 0 must be finite (no NaN from the q-p power).
    z = torch.zeros(3)
    finite = bool(torch.isfinite(rho(z, 0.02, P, Q_EXP, 1.0)).all()
                  and torch.isfinite(rho_prime(z, 0.02, P, Q_EXP, 1.0)).all())
    const = torch.full((16, 16), 0.37)
    bt_const = surrogate_weights(const, 0.02, P, Q_EXP, 1.0)
    g_const = true_gradient(const, const, a, SIGMA_W, 0.02, P, Q_EXP, 1.0)
    finite = finite and bool(torch.isfinite(bt_const).all() and torch.isfinite(g_const).all())
    interior_sum = bt_const.sum(0)[5, 5].item()
    print(f"  rho, rho_prime, b_tilde, autograd gradient finite at Delta = 0: {finite}")
    print(f"  constant image: sum_r b_tilde at interior pixel = {interior_sum:.6f} "
          f"(eq. 15 gives 1/(2 sigma_x^2) = {1 / (2 * 0.02 ** 2):.6f})")
    out["finite_at_zero"] = finite

    # 3. b_tilde (14) equals b rho'(D)/(2D) of (12), and is continuous at the switch to (15).
    dd = torch.tensor([1e-6, 1e-4, 0.01, 0.05, 0.3, -0.2])
    img = torch.zeros(3, 3)
    errs = []
    for val in dd:
        img = torch.zeros(3, 3)
        img[1, 1] = val                      # Delta = val for every neighbor of the center
        bt = surrogate_weights(img, 0.02, P, Q_EXP, 1.0)[:, 1, 1]
        ref = torch.tensor(B_WEIGHTS) * rho_prime(val, 0.02, P, Q_EXP, 1.0) / (2 * val)
        errs.append(((bt - ref).abs().max() / ref.abs().max()).item())
    small = torch.zeros(3, 3)
    small[1, 1] = 1.0001 * EPS
    just_above = surrogate_weights(small, 0.02, P, Q_EXP, 1.0)[4, 1, 1].item()
    at_zero = surrogate_weights(torch.zeros(3, 3), 0.02, P, Q_EXP, 1.0)[4, 1, 1].item()
    jump = abs(just_above - at_zero) / at_zero
    print(f"  b_tilde (14) vs b*rho'(D)/(2D) (12): max rel err = {max(errs):.3e}")
    print(f"  b_tilde relative jump across the EPS switch to (15): {jump:.3e}")
    out["b_tilde_vs_eq12_max_rel"] = max(errs)
    out["b_tilde_switch_jump_rel"] = jump

    # 4. Autograd true gradient against the analytic gradient (16), full image.
    torch.manual_seed(SEED + 1)
    probe = y + 0.05 * torch.randn_like(y)
    for name, xx, sx, T in [("x=y, R2", y, 0.020, 1.0), ("x=y, R1", y, 0.036, 100.0),
                            ("x=x_true, R2", x_true, 0.020, 1.0),
                            ("x=y+noise, R3", probe, 0.010, 1.0)]:
        g_auto = true_gradient(xx, y, a, SIGMA_W, sx, P, Q_EXP, T)
        g_16 = eq16_gradient(xx, y, a, sx, T)
        err = ((g_auto - g_16).abs().max() / g_16.abs().max()).item()
        out[f"grad_vs_eq16_{name}"] = err
        print(f"  autograd gradient vs eq. (16), {name}: max rel err = {err:.3e}")

    # 5. g^t H g of (18) against the second difference of the quadratic Q of (13).
    for name, xx, sx, T in [("R2", y, 0.020, 1.0), ("R1", y, 0.036, 100.0)]:
        g = true_gradient(xx, y, a, SIGMA_W, sx, P, Q_EXP, T)
        alpha = optimal_step_size(xx, g, a, SIGMA_W, sx, P, Q_EXP, T)
        gHg = (g * g).sum() / alpha
        t = 1e-3
        second = (surrogate_cost(xx + t * g, xx, y, a, sx, T)
                  + surrogate_cost(xx - t * g, xx, y, a, sx, T)
                  - 2 * surrogate_cost(xx, xx, y, a, sx, T)) / t ** 2
        err = ((gHg - second).abs() / gHg).item()
        out[f"gHg_vs_Q_second_diff_{name}"] = err
        print(f"  g^t H g (18) vs second difference of Q (13), {name}: rel err = {err:.3e}")
        # Surrogate and true gradient agree at x' (tangency).
        xv = xx.detach().clone().requires_grad_(True)
        gq = torch.autograd.grad(surrogate_cost(xv, xx, y, a, sx, T), xv)[0]
        err = ((gq - g).abs().max() / g.abs().max()).item()
        out[f"grad_Q_vs_grad_f_{name}"] = err
        print(f"  grad Q(x; x) vs grad f(x), {name}: max rel err = {err:.3e}")

    # 6. 1-D majorization check for D3.
    dg = torch.linspace(-0.3, 0.3, 60001)
    for dp in [0.02, 0.15]:
        dpt = torch.tensor(dp)
        sur = (rho_prime(dpt, 0.2, P, Q_EXP, 1.0) / (2 * dpt) * (dg ** 2 - dpt ** 2)
               + rho(dpt, 0.2, P, Q_EXP, 1.0))
        gap = sur - rho(dg, 0.2, P, Q_EXP, 1.0)
        at_pts = [(rho_prime(dpt, 0.2, P, Q_EXP, 1.0) / (2 * dpt) * (s ** 2 - dpt ** 2)
                   + rho(dpt, 0.2, P, Q_EXP, 1.0) - rho(s, 0.2, P, Q_EXP, 1.0)).item()
                  for s in (dpt, -dpt)]
        out[f"surrogate_min_gap_{dp}"] = gap.min().item()
        print(f"  surrogate at Delta'={dp}: min(surrogate - rho) on grid = {gap.min().item():.3e}, "
              f"gap at +/-Delta' = {at_pts[0]:.1e}, {at_pts[1]:.1e}")
    return out


# --------------------------------------------------------------------------
def plot_d1_d3():
    sx, T = 0.2, 1.0
    d = torch.linspace(-0.3, 0.3, 1201)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(d, gaussian_rho(d, sx), color=C_BLUE, label="Quadratic, eq. (2)")
    ax1.plot(d, rho(d, sx, P, Q_EXP, T), color=C_ORANGE, label="QGGMRF, eq. (3)")
    ax1.set_xlabel("Δ"); ax1.set_ylabel("ρ(Δ)"); ax1.set_title("Potential")
    ax1.legend()
    ax2.plot(d, d / sx ** 2, color=C_BLUE, label="Quadratic, Δ/σx²")
    ax2.plot(d, rho_prime(d, sx, P, Q_EXP, T), color=C_ORANGE, label="QGGMRF, eq. (4)")
    ax2.set_xlabel("Δ"); ax2.set_ylabel("ρ′(Δ)"); ax2.set_title("Influence function")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.axvline(T * sx, color=C_MUTED, lw=0.8, ls=":")
        ax.axvline(-T * sx, color=C_MUTED, lw=0.8, ls=":")
    fig.suptitle(f"p = {P}, q = {Q_EXP}, T = {T}, σx = {sx}   (dotted: |Δ| = Tσx)", color=C_INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D1_potential_influence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(d, rho(d, sx, P, Q_EXP, T), color=C_INK, label="QGGMRF ρ(Δ)")
    for dp, col in [(0.02, C_BLUE), (0.15, C_ORANGE)]:
        dpt = torch.tensor(dp)
        r0 = rho(dpt, sx, P, Q_EXP, T)
        sur = rho_prime(dpt, sx, P, Q_EXP, T) / (2 * dpt) * (d ** 2 - dpt ** 2) + r0
        ax.plot(d, sur, color=col, ls="--", label=f"surrogate, Δ′ = {dp}")
        ax.plot([-dp, dp], [r0, r0], "o", ms=8, color=col, mec="white", mew=1.5)
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel("Δ"); ax.set_ylabel("value")
    ax.set_title(f"Symmetric bound surrogates (p={P}, q={Q_EXP}, T={T}, σx={sx})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D3_surrogates.png"), dpi=150)
    plt.close(fig)
    print("saved results/D1_potential_influence.png, results/D3_surrogates.png")


# --------------------------------------------------------------------------
def run_or_load(name, cfg, y, a, y_sum):
    path = os.path.join(RUN_DIR, f"{name}.pt")
    if os.path.exists(path):
        saved = torch.load(path)
        same = (saved["sigma_x"] == cfg["sigma_x"] and saved["T"] == cfg["T"]
                and saved["num_iters"] == NUM_ITERS and saved["omega"] == OMEGA
                and saved["y_checksum"] == y_sum)
        if same:
            print(f"[{name}] loaded cached run from {path}")
            return saved
        print(f"[{name}] cached run has different settings; rerunning")
    print(f"[{name}] running {cfg['prior']}, T={cfg['T']}, sigma_x={cfg['sigma_x']}, "
          f"N={NUM_ITERS}, omega={OMEGA} ...", flush=True)
    t0 = time.perf_counter()
    x_hat, hist = reconstruct(y, a, SIGMA_W, cfg["sigma_x"], P, Q_EXP, cfg["T"],
                              NUM_ITERS, OMEGA)
    elapsed = time.perf_counter() - t0
    saved = dict(name=name, prior=cfg["prior"], T=cfg["T"], sigma_x=cfg["sigma_x"],
                 num_iters=NUM_ITERS, omega=OMEGA, y_checksum=y_sum,
                 x_hat=x_hat, cost_history=hist, time_s=elapsed)
    torch.save(saved, path)
    print(f"[{name}] done in {elapsed:.2f} s, f(y) = {hist[0]:.6e} -> "
          f"f(x_50) = {hist[-1]:.6e}; saved {path}", flush=True)
    return saved


def monotone_report(hist):
    h = torch.tensor(hist)
    diffs = h[1:] - h[:-1]
    return dict(monotone=bool((diffs < 0).all()), num_increases=int((diffs >= 0).sum()),
                largest_change=diffs.max().item())


def show(ax, img, title):
    ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)


def figures(x_true, y, a, runs):
    import matplotlib.patches as mpatches

    r1, r2 = runs["R1"], runs["R2"]
    panels = [("x_true", x_true), ("y (blurred + noise)", y),
              (f"R1 Gaussian MRF, σx={r1['sigma_x']}", r1["x_hat"]),
              (f"R2 QGGMRF, σx={r2['sigma_x']}", r2["x_hat"])]
    rc = f"rows {CROP_R0}–{CROP_R0 + CROP - 1}, cols {CROP_C0}–{CROP_C0 + CROP - 1}"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (t, img) in zip(axes.flat, panels):
        show(ax, img, t)
        ax.add_patch(mpatches.Rectangle((CROP_C0 - 0.5, CROP_R0 - 0.5), CROP, CROP,
                                        fill=False, ec="#e34948", lw=1.2))
    fig.suptitle(f"Full images, gray scale [0, 1] (red box = crop, {rc})")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D6_full_images.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (t, img) in zip(axes, panels):
        show(ax, crop(img), t)
    fig.suptitle(f"128×128 crop: {rc}; gray scale [0, 1]")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D6_crops.png"), dpi=150)
    plt.close(fig)

    # Individual image files (display range [0, 1]; data itself is not clipped).
    for tag, img in [("x_true", x_true), ("y", y)] + [(k, v["x_hat"]) for k, v in runs.items()]:
        plt.imsave(os.path.join(RES, "images", f"{tag}_full.png"), img.numpy(),
                   cmap="gray", vmin=0, vmax=1)
        plt.imsave(os.path.join(RES, "images", f"{tag}_crop.png"), crop(img).numpy(),
                   cmap="gray", vmin=0, vmax=1)

    # D7 edge profile, with a zoom on the beak's left edge.
    cols = torch.arange(CROP_C0, CROP_C0 + CROP)
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    for lab, img, col in [("x_true", x_true, C_INK), ("R1 Gaussian MRF", r1["x_hat"], C_BLUE),
                          ("R2 QGGMRF", r2["x_hat"], C_ORANGE)]:
        prof = img[PROFILE_ROW, CROP_C0:CROP_C0 + CROP]
        lw = 1.5 if lab == "x_true" else 2.0
        ax.plot(cols, prof, color=col, label=lab, lw=lw)
        axz.plot(cols, prof, color=col, label=lab, lw=lw, marker="o", ms=4)
    ax.axvspan(EDGE_COLS[0], EDGE_COLS[1], color=C_GRID, alpha=0.6, lw=0)
    ax.set_xlabel("column"); ax.set_ylabel("pixel value")
    ax.set_title(f"Edge profile along row {PROFILE_ROW}, columns {CROP_C0}–{CROP_C0 + CROP - 1}")
    ax.legend()
    axz.set_xlim(EDGE_COLS[0], EDGE_COLS[1])
    axz.set_xlabel("column"); axz.set_title("Zoom: beak's left edge (shaded)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D7_edge_profile.png"), dpi=150)
    plt.close(fig)

    # Profile line marked on the crop.
    fig, ax = plt.subplots(figsize=(4, 4))
    show(ax, crop(x_true), f"x_true crop, profile row {PROFILE_ROW}")
    ax.axhline(PROFILE_ROW - CROP_R0, color="#e34948", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D7_profile_location.png"), dpi=150)
    plt.close(fig)

    # D8 sigma_x study.
    fig, axes = plt.subplots(2, 3, figsize=(11, 8))
    for row, names in enumerate([("R3", "R2", "R4"), ("R5", "R1", "R6")]):
        for ax, n, lab in zip(axes[row], names, ("center / 2", "center", "center × 2")):
            r = runs[n]
            show(ax, crop(r["x_hat"]),
                 f"{n} {r['prior']}, σx={r['sigma_x']} ({lab})\nRMSE = {r['rmse']:.5f}")
    fig.suptitle(f"σx study, crop {rc}, gray scale [0, 1]")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(os.path.join(RES, "D8_sigma_study.png"), dpi=150)
    plt.close(fig)

    # D9 summed surrogate weights at the final R2 image.
    bt = surrogate_weights(r2["x_hat"], r2["sigma_x"], P, Q_EXP, r2["T"]).sum(0)
    vmax = 1 / (2 * r2["sigma_x"] ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for ax, img, t in [(axes[0], bt, "full image"), (axes[1], crop(bt), "crop")]:
        im = ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Σr b̃s,r at final R2 image, {t}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.suptitle(f"Display range linear [0, {vmax:.0f}] = [0, 1/(2σx²)]; data range "
                 f"[{bt.min().item():.3g}, {bt.max().item():.4g}]")
    fig.savefig(os.path.join(RES, "D9_surrogate_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved D6, D7, D8, D9 figures and results/images/*.png")
    return dict(d9_vmax=vmax, d9_min=bt.min().item(), d9_max=bt.max().item(),
                d9_interior_median=bt[1:-1, 1:-1].median().item())


def plot_d5(hist, name):
    k = list(range(len(hist)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(k, hist, color=C_BLUE, marker="o", ms=3)
    ax1.set_xlabel("iteration k"); ax1.set_ylabel("true cost f(x⁽ᵏ⁾)")
    ax1.set_title("True cost vs. iteration")
    h = torch.tensor(hist)
    dec = (h[:-1] - h[1:])
    ax2.semilogy(k[1:], dec.clamp_min(1e-300), color=C_BLUE, marker="o", ms=3)
    ax2.set_xlabel("iteration k"); ax2.set_ylabel("f(x⁽ᵏ⁻¹⁾) − f(x⁽ᵏ⁾)")
    ax2.set_title("Decrease per iteration (log scale, all > 0)")
    fig.suptitle(f"Baseline QGGMRF ({name}): σw={SIGMA_W}, σx=0.02, T=1, p={P}, q={Q_EXP}, "
                 f"N={NUM_ITERS}, ω={OMEGA}, seed {SEED}")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "D5_cost_vs_iteration.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(os.path.join(RES, "images"), exist_ok=True)
    sys.stdout = Tee(os.path.join(RES, "run_log.txt"))
    print(f"\n######## run_lab4.py  {time.strftime('%Y-%m-%d %H:%M:%S')} ########")
    print(f"device={DEVICE}, dtype={torch.get_default_dtype()}, torch {torch.__version__}, "
          f"threads={torch.get_num_threads()}")

    x_true = load_pgm("kodim23.pgm").to(DEVICE)
    a = load_kernel("levin09_kernels/levin09_kernel1.txt").to(DEVICE)

    # One noisy image for every run; saved so later reruns use the identical y.
    y_path = os.path.join(RES, "y_noisy.pt")
    torch.manual_seed(SEED)
    w = torch.randn(x_true.shape, dtype=torch.float64, device=DEVICE) * SIGMA_W
    y = forward(x_true, a) + w            # not clipped
    if os.path.exists(y_path):
        y_saved = torch.load(y_path)
        assert torch.equal(y_saved, y), "saved y differs from regenerated y"
    else:
        torch.save(y, y_path)
    y_sum = float(y.sum().item())
    print(f"seed={SEED}, sigma_w={SIGMA_W}, kernel 1, image {tuple(x_true.shape)}, "
          f"y range [{y.min().item():.4f}, {y.max().item():.4f}] (unclipped), "
          f"RMSE(y, x_true)={rmse(y, x_true):.6f}")

    checks = run_checks(x_true, y, a)
    plot_d1_d3()

    # ---- Task 6a: baseline QGGMRF monotonicity check (same settings as R2) ----
    print("\n=== Baseline monotonicity check (QGGMRF, sigma_x=0.020, T=1) ===")
    base = run_or_load("R2", RUNS["R2"], y, a, y_sum)
    mono = monotone_report(base["cost_history"])
    plot_d5(base["cost_history"], "same settings as R2")
    print(f"monotone decreasing at every iteration: {mono['monotone']}  "
          f"(increases: {mono['num_increases']}, largest change "
          f"f(k+1)-f(k) = {mono['largest_change']:.3e})")
    print("saved results/D5_cost_vs_iteration.png")
    if not mono["monotone"]:
        raise SystemExit("Baseline cost increased; fix surrogate weights / step size first.")

    # ---- Step 7: runs R1-R6 ----
    print("\n=== Runs R1-R6 ===")
    runs = {}
    for name, cfg in RUNS.items():
        r = run_or_load(name, cfg, y, a, y_sum)
        r["rmse"] = rmse(r["x_hat"], x_true)
        r["mono"] = monotone_report(r["cost_history"])
        runs[name] = r

    print(f"\n{'run':<4}{'prior':<14}{'T':>6}{'sigma_x':>9}{'final true cost':>18}"
          f"{'RMSE':>11}{'time (s)':>10}{'monotone':>10}")
    with open(os.path.join(RES, "D10_table.csv"), "w") as f:
        f.write("run,prior,T,sigma_x,final_true_cost,rmse,wall_clock_s,monotone\n")
        for n, r in runs.items():
            print(f"{n:<4}{r['prior']:<14}{r['T']:>6g}{r['sigma_x']:>9.3f}"
                  f"{r['cost_history'][-1]:>18.6e}{r['rmse']:>11.6f}{r['time_s']:>10.2f}"
                  f"{str(r['mono']['monotone']):>10}")
            f.write(f"{n},{r['prior']},{r['T']:g},{r['sigma_x']},{r['cost_history'][-1]:.9e},"
                    f"{r['rmse']:.6f},{r['time_s']:.3f},{r['mono']['monotone']}\n")
    print("saved results/D10_table.csv")

    extra = figures(x_true, y, a, runs)

    print(f"\nEdge (cols {EDGE_COLS[0]}-{EDGE_COLS[1]}) and flat-patch statistics:")
    # Edge sharpness: steepest adjacent-pixel step across the beak's left edge,
    # on the profile row and averaged over rows 200-260; flat-region noise as
    # the std of a flat background patch.
    e0, e1 = EDGE_COLS
    edge_stats = {}
    for k, img in [("x_true", x_true)] + [(n, r["x_hat"]) for n, r in runs.items()]:
        steps = (img[:, e0 + 1:e1 + 1] - img[:, e0:e1]).abs()
        edge_stats[k] = dict(
            profile_row_max_step=steps[PROFILE_ROW].max().item(),
            rows_200_260_mean_max_step=steps[200:261].max(1).values.mean().item(),
            flat_patch_std=img[FLAT_BOX].std().item())
        print(f"  {k:<7} max step on row {PROFILE_ROW}: "
              f"{edge_stats[k]['profile_row_max_step']:.3f}   mean max step rows 200-260: "
              f"{edge_stats[k]['rows_200_260_mean_max_step']:.3f}   flat patch std: "
              f"{edge_stats[k]['flat_patch_std']:.4f}")
    crop_rmse = {k: rmse(crop(r["x_hat"]), crop(x_true)) for k, r in runs.items()}
    print(f"crop RMSE: { {k: round(v, 6) for k, v in crop_rmse.items()} }")

    summary = dict(
        seed=SEED, sigma_w=SIGMA_W, p=P, q=Q_EXP, num_iters=NUM_ITERS, omega=OMEGA,
        crop=dict(row0=CROP_R0, col0=CROP_C0, size=CROP), profile_row=PROFILE_ROW,
        y_rmse=rmse(y, x_true), y_crop_rmse=rmse(crop(y), crop(x_true)),
        edge_cols=EDGE_COLS, flat_box=[[250, 290], [385, 400]],
        checks=checks, baseline_monotone=mono, edge_stats=edge_stats, crop_rmse=crop_rmse,
        runs={n: dict(prior=r["prior"], T=r["T"], sigma_x=r["sigma_x"],
                      f_y=r["cost_history"][0], final_cost=r["cost_history"][-1],
                      rmse=r["rmse"], time_s=r["time_s"], monotone=r["mono"])
              for n, r in runs.items()},
        **extra,
    )
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/summary.json")


if __name__ == "__main__":
    main()

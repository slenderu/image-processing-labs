import torch
import torch.nn.functional as F

from lab3 import forward


NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             ( 0, -1),          ( 0, 1),
             ( 1, -1), ( 1, 0), ( 1, 1)]

B_WEIGHTS = [1/12, 1/6, 1/12,
             1/6,        1/6,
             1/12, 1/6, 1/12]

# Floor on |Delta| before the power |Delta/(T sigma_x)|^(q-p), and the
# threshold at or below which the surrogate weight switches to eq. (15).
EPS = 1e-8


def rho(delta, sigma_x, p, q, T):
    """Return the QGGMRF potential of equation (3), elementwise.

    |delta| is floored at EPS only inside the power |delta/(T sigma_x)|^(q-p),
    which is infinite at delta = 0 because q < p.  The leading |delta|^p is
    not floored, so rho(0) = 0 and the value and its autograd gradient stay
    finite and accurate near 0.

    delta    float64 tensor of any shape
    sigma_x  float, scale of the pixel differences
    p, q     float, exponents near zero and for large differences
    T        float, transition threshold in units of sigma_x
    returns  float64 tensor of the same shape
    """
    abs_d = delta.abs()
    u = (abs_d.clamp_min(EPS) / (T * sigma_x)) ** (q - p)
    return abs_d ** p / (p * sigma_x ** p) * (u / (1 + u))


def rho_prime(delta, sigma_x, p, q, T):
    """Return the QGGMRF influence function of equation (4), elementwise.

    |delta| is floored at EPS only inside the (q-p) power, as in rho, so
    rho_prime(delta) -> delta / sigma_x^2 as delta -> 0 and rho_prime(0) = 0.

    delta    float64 tensor of any shape
    sigma_x  float, scale of the pixel differences
    p, q     float, exponents near zero and for large differences
    T        float, transition threshold in units of sigma_x
    returns  float64 tensor of the same shape
    """
    abs_d = delta.abs()
    u = (abs_d.clamp_min(EPS) / (T * sigma_x)) ** (q - p)
    return (abs_d ** (p - 1) / sigma_x ** p
            * u * (q / p + u) / (1 + u) ** 2
            * torch.sign(delta))


def neighbor_diffs(x):
    """Return the differences x[s] - x[r] for the 8 neighbor offsets.

    Plane i holds x minus x shifted by NEIGHBORS[i].  Entries whose
    neighbor falls outside the image are set to zero, and the matching
    entries of the mask are zero, so that a pair counts only when both
    pixels are inside.

    x        (H, W) float64 tensor
    returns  (diffs, valid), each (8, H, W), float64 and bool
    """
    H, W = x.shape
    x_pad = F.pad(x[None, None], (1, 1, 1, 1))[0, 0]
    inside = F.pad(torch.ones_like(x)[None, None], (1, 1, 1, 1))[0, 0] > 0
    neighbors = torch.stack([x_pad[1 + di:1 + di + H, 1 + dj:1 + dj + W]
                             for di, dj in NEIGHBORS])
    valid = torch.stack([inside[1 + di:1 + di + H, 1 + dj:1 + dj + W]
                         for di, dj in NEIGHBORS])
    diffs = torch.where(valid, x[None] - neighbors, torch.zeros_like(neighbors))
    return diffs, valid


def surrogate_weights(x, sigma_x, p, q, T):
    """Return the b_tilde coefficients of equations (14) and (15).

    Used inside optimal_step_size to form the surrogate Hessian.  Where
    |Delta'| <= EPS the limit (15), b / (2 sigma_x^2), is used; this limit
    assumes p = 2.  |Delta'| is floored at EPS before the power in (14) so
    the unused branch of torch.where holds no inf or NaN.

    x        (H, W) float64 tensor, the point of approximation x'
    sigma_x  float
    p, q, T  float, QGGMRF parameters (p must be 2)
    returns  (8, H, W) float64 tensor, zero where the pair is invalid
    """
    diffs, valid = neighbor_diffs(x)
    b = torch.tensor(B_WEIGHTS, dtype=x.dtype, device=x.device)[:, None, None]
    abs_d = diffs.abs()
    abs_d_safe = abs_d.clamp_min(EPS)
    u = (abs_d_safe / (T * sigma_x)) ** (q - p)
    w14 = (b * abs_d_safe ** (p - 2) / (2 * sigma_x ** p)
           * u * (q / p + u) / (1 + u) ** 2)
    w15 = b / (2 * sigma_x ** 2) * torch.ones_like(abs_d)
    b_tilde = torch.where(abs_d <= EPS, w15, w14)
    return torch.where(valid, b_tilde, torch.zeros_like(b_tilde))


def true_cost(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return the true cost f(x) of equation (1).

    Differentiable in x, so torch.autograd can take its gradient.  Floor
    |x_s - x_r| at a small value (for example 1e-8) before the QGGMRF
    power: rho is NaN at Delta = 0 in the forward value too, not only in
    autograd, because |Delta/(T sigma_x)|^(q-p) is infinite there.  Apply
    the neighbor validity mask after rho so floored border pairs contribute
    exactly zero.

    x, y     (H, W) float64 tensors, the image and the data
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w  float, noise standard deviation assumed by the data term
    sigma_x, p, q, T  float, QGGMRF parameters
    returns  float64 scalar tensor
    """
    data_term = ((y - forward(x, a)) ** 2).sum() / (2 * sigma_w ** 2)
    diffs, valid = neighbor_diffs(x)
    b = torch.tensor(B_WEIGHTS, dtype=x.dtype, device=x.device)[:, None, None]
    pot = rho(diffs, sigma_x, p, q, T)
    pot = torch.where(valid, pot, torch.zeros_like(pot))
    prior_term = 0.5 * (b * pot).sum()
    return data_term + prior_term


def true_gradient(x, y, a, sigma_w, sigma_x, p, q, T):
    """Return the gradient of the true cost, equation (16).

    Computed with torch.autograd.grad(true_cost(x, ...), x).  x must have
    requires_grad set, and the forward and adjoint of a must be torch
    operations so the graph reaches x.  Equation (16) is the analytic form
    to check it against.  A detached copy of x with requires_grad set is
    made here, so the caller's x is not modified.

    x, y     (H, W) float64 tensors
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w, sigma_x, p, q, T  float
    returns  (H, W) float64 tensor
    """
    x_var = x.detach().clone().requires_grad_(True)
    f = true_cost(x_var, y, a, sigma_w, sigma_x, p, q, T)
    return torch.autograd.grad(f, x_var)[0]


def optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T):
    """Return the exact surrogate step size alpha* of equations (17)-(18).

    Builds the surrogate weights at x and uses one projection A g to form
    the denominator g^t H g.  g is the gradient from true_gradient.  The
    denominator uses only A g, the surrogate weights at x, and sigma_w; it
    does not use y.

    x        (H, W) float64 tensor, the point of approximation
    g        (H, W) float64 tensor, the true gradient at x
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w, sigma_x, p, q, T  float
    returns  float64 scalar tensor
    """
    b_tilde = surrogate_weights(x, sigma_x, p, q, T)
    g_diffs, _ = neighbor_diffs(g)
    gHg = ((forward(g, a) ** 2).sum() / sigma_w ** 2
           + (b_tilde * g_diffs ** 2).sum())
    return (g * g).sum() / gHg


def reconstruct(y, a, sigma_w, sigma_x, p, q, T, num_iters, omega):
    """Steepest descent with the surrogate step size, starting at y.

    Each iteration: g = true_gradient(x); alpha = optimal_step_size(x, g);
    x = x - omega * alpha * g.  No clipping or nonnegativity is applied.

    y        (H, W) float64 tensor, the data
    a        (Ka, Kb) float64 tensor, the blur kernel
    sigma_w, sigma_x, p, q, T  float
    num_iters  int, number of iterations N
    omega    float in (0, 2), over-relaxation factor
    returns  (x, cost_history), the final image and a list of the true
             cost after each iteration, of length num_iters + 1 whose
             first entry is f(y)
    """
    x = y.detach().clone()
    with torch.no_grad():
        cost_history = [true_cost(x, y, a, sigma_w, sigma_x, p, q, T).item()]
    for _ in range(num_iters):
        g = true_gradient(x, y, a, sigma_w, sigma_x, p, q, T)
        with torch.no_grad():
            alpha = optimal_step_size(x, g, a, sigma_w, sigma_x, p, q, T)
            x = x - omega * alpha * g
            cost_history.append(true_cost(x, y, a, sigma_w, sigma_x, p, q, T).item())
    return x, cost_history

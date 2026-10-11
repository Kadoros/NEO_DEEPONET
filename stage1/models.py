"""Part 2 models.  Every model maps a batch of shapes to û of shape (B, T, N_IC, N).

Batch dict (all torch tensors on one device, B shapes):
  x (B,N,3)  m_pc (B,N)  u0 (B,I,N)  t (B,T)
  Phi (B,N,K)  lam (B,K)  coef (B,K,I) = Φᵀ M_pc u0       -- for the chosen basis
  F (B,N,256)  z (B,256)  coef_neo (B,K,I)                  -- NEO features / NEO-basis coefficients
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


def mlp(sizes, act=nn.SiLU, last_zero=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if last_zero:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


def log_t_feature(t):
    return (torch.log(t) + 3.0) / 3.0          # t ∈ [~5e-4, ~4]  ->  roughly [-1.5, 1.5]


class Galerkin(nn.Module):
    """G-*: û = Φ exp(-tΛ) Φᵀ M u0 (no parameters)."""

    def forward(self, b):
        g = torch.exp(-b["lam"][:, None, :] * b["t"][:, :, None])              # (B,T,K)
        return torch.einsum("bnk,btk,bki->btin", b["Phi"], g, b["coef"])


class SpectralMultiplier(nn.Module):
    """g_θ(λ_k t, λ_k/λ_1, z) shared by all modes;  g = exp(-s·ρ),  ρ = softplus(MLP) (ρ ≡ 1 at init).

    Inputs to the MLP: log s, log(λ_k/λ_1), and a 16-d linear projection of z.
    Mode 0 (λ ≈ 0, the mean) is fixed to g = 1 (mass conservation).
    """

    def __init__(self, z_dim=256, z_proj=16, hidden=64):
        super().__init__()
        self.zp = nn.Linear(z_dim, z_proj)
        self.net = mlp([2 + z_proj, hidden, hidden, 1], last_zero=True)
        with torch.no_grad():
            self.net[-1].bias.fill_(math.log(math.e - 1))                    # softplus^{-1}(1)

    def forward(self, lam, t, z):
        s = lam[:, None, 1:] * t[:, :, None]                                   # (B,T,K-1)
        r = lam[:, 1:] / lam[:, 1:2]                                           # (B,K-1)
        B, T, K1 = s.shape
        feats = torch.cat([torch.log(s)[..., None] / 3.0,
                           torch.log(r)[:, None, :, None].expand(B, T, K1, 1),
                           self.zp(z)[:, None, None, :].expand(B, T, K1, -1)], -1)
        rho = Fn.softplus(self.net(feats)[..., 0])
        g = torch.exp(-s * rho)
        return torch.cat([torch.ones_like(g[..., :1]), g], -1)                # (B,T,K)


class Correction(nn.Module):
    """ε_θ(x, t): DeepONet with branch = MLP(NEO-basis coefficients of u0),
    trunk = MLP([F(x), x, log t]).  Output layer of the trunk starts at zero."""

    def __init__(self, k=96, f_dim=256, p=64, hidden=128):
        super().__init__()
        self.branch = mlp([k, hidden, hidden, p])
        self.trunk = mlp([f_dim + 4, hidden, hidden, p], last_zero=True)
        self.bias = nn.Parameter(torch.zeros(()))
        self.coef_scale = 1 / math.sqrt(4 * math.pi)

    def forward(self, b):
        B, N, _ = b["x"].shape
        T = b["t"].shape[1]
        br = self.branch(b["coef_neo"].transpose(1, 2) * self.coef_scale)          # (B,I,p)
        lt = log_t_feature(b["t"])[:, :, None, None].expand(B, T, N, 1)
        tin = torch.cat([b["F"][:, None].expand(B, T, N, -1), b["x"][:, None].expand(B, T, N, 3), lt], -1)
        tr = self.trunk(tin)                                                        # (B,T,N,p)
        return torch.einsum("bip,btnp->btin", br, tr) + self.bias


class LearnedSpectral(nn.Module):
    """L-*: û = Σ_k g_θ(λ_k t, λ_k/λ_1, z) ⟨u0, φ_k⟩_M φ_k + ε_θ(x, t)."""

    def __init__(self, use_eps=False):
        super().__init__()
        self.g = SpectralMultiplier()
        self.eps = Correction() if use_eps else None

    def forward(self, b):
        g = self.g(b["lam"], b["t"], b["z"])
        u = torch.einsum("bnk,btk,bki->btin", b["Phi"], g, b["coef"])
        return u if self.eps is None else u + self.eps(b)


class DeepONetBase(nn.Module):
    """DON-base: branch = PointNet([x, u0]) (mass-weighted mean + max pooling), trunk = MLP([x, log t])."""

    def __init__(self, p=128, hidden=128):
        super().__init__()
        self.point = mlp([4, 64, 128, 256])
        self.head = mlp([512, 256, p])
        self.trunk = mlp([4, hidden, hidden, hidden, p])
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, b):
        B, N, _ = b["x"].shape
        I, T = b["u0"].shape[1], b["t"].shape[1]
        h = self.point(torch.cat([b["x"][:, None].expand(B, I, N, 3), b["u0"][..., None]], -1))   # (B,I,N,256)
        w = (b["m_pc"] / b["m_pc"].sum(1, keepdim=True))[:, None, :, None]
        br = self.head(torch.cat([(w * h).sum(2), h.amax(2)], -1))                               # (B,I,p)
        lt = log_t_feature(b["t"])[:, :, None, None].expand(B, T, N, 1)
        tr = self.trunk(torch.cat([b["x"][:, None].expand(B, T, N, 3), lt], -1))                 # (B,T,N,p)
        return torch.einsum("bip,btnp->btin", br, tr) + self.bias


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------------------------------------------------------------------
# ε-path v2 (REPORT_eps): residual inputs + physics prior
#   batch additionally carries  r (B,I,N) = u0 − Y Yᵀ M u0  and  rp (B,T,I,N) = (M + tL)^{-1} M r
# ------------------------------------------------------------------------------------------------
class ResidualEps(nn.Module):
    """ε_θ(x, t) = Σ_p branch_p · trunk_p(x, t).

    branch: mass-weighted cross-attention encoder over point tokens (x, u0, r): n_q learned
            queries attend to all points with softmax weights ∝ m_i·exp(q·k_i) (log m added to
            the logits = area-weighted quadrature), then flatten -> MLP -> p.
    trunk : pointwise [r(x), r_p(x,t) (optional), F(x), x, log t] -> MLP -> p.  The IC-independent
            part [F, x, log t] is embedded once and added to the embedding of [r, r_p] before the
            non-linearity.  Last trunk layer starts at zero (ε ≡ 0 at init).
    """

    def __init__(self, use_rp=True, r_scale=1.0, f_dim=256, p=64, hidden=128, d=64, n_q=8, heads=4):
        super().__init__()
        self.use_rp, self.r_scale, self.heads = use_rp, r_scale, heads
        self.embed = mlp([5, d, d])
        self.queries = nn.Parameter(torch.randn(n_q, d) / math.sqrt(d))
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.head = mlp([n_q * d, 256, p])
        self.trunk_shared = nn.Linear(f_dim + 4, hidden)
        self.trunk_ic = nn.Linear(2 if use_rp else 1, hidden, bias=False)
        self.trunk_rest = mlp([hidden, hidden, p], last_zero=True)
        self.bias = nn.Parameter(torch.zeros(()))

    def branch(self, b):
        x, u0, m = b["x"], b["u0"], b["m_pc"]
        B, I, N = u0.shape
        tok = self.embed(torch.cat([x[:, None].expand(B, I, N, 3), u0[..., None],
                                    (b["r"] * self.r_scale)[..., None]], -1)).reshape(B * I, N, -1)
        d = tok.shape[-1]
        H, dh = self.heads, d // self.heads
        q = self.wq(self.queries).view(1, -1, H, dh).transpose(1, 2).expand(B * I, H, -1, dh)
        k = self.wk(tok).view(B * I, N, H, dh).transpose(1, 2)
        v = self.wv(tok).view(B * I, N, H, dh).transpose(1, 2)
        logm = torch.log(m / m.sum(1, keepdim=True)).repeat_interleave(I, 0)[:, None, None, :]
        o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=logm.to(q.dtype))       # (B·I,H,Q,dh)
        o = self.wo(o.transpose(1, 2).reshape(B * I, -1, d))
        return self.head(o.reshape(B, I, -1))                                              # (B,I,p)

    def forward(self, b):
        B, N, _ = b["x"].shape
        T, I = b["t"].shape[1], b["u0"].shape[1]
        lt = log_t_feature(b["t"])[:, :, None, None].expand(B, T, N, 1)
        hs = self.trunk_shared(torch.cat([b["F"][:, None].expand(B, T, N, -1),
                                          b["x"][:, None].expand(B, T, N, 3), lt], -1))       # (B,T,N,h)
        ic = [(b["r"] * self.r_scale)[:, None].expand(B, T, I, N)]
        if self.use_rp:
            ic.append(b["rp"] * self.r_scale)
        h = Fn.silu(hs[:, :, None] + self.trunk_ic(torch.stack(ic, -1)))                   # (B,T,I,N,h)
        tr = self.trunk_rest(h)                                                             # (B,T,I,N,p)
        return torch.einsum("bip,btinp->btin", self.branch(b), tr) + self.bias


class LNeoResidual(nn.Module):
    """û = Σ_k g_θ(λ_k t, λ_k/λ_1, z) ⟨u0, φ_k⟩_M φ_k  [+ r_p(t)]  [+ ε_θ]   (NEO basis).

    eps: None | "rp" (inputs r and r_p) | "norp" (input r only)."""

    def __init__(self, add_rp=False, eps=None, r_scale=1.0):
        super().__init__()
        self.g = SpectralMultiplier()
        self.add_rp = add_rp
        self.eps = None if eps is None else ResidualEps(use_rp=(eps == "rp"), r_scale=r_scale)

    def forward(self, b):
        u = torch.einsum("bnk,btk,bki->btin", b["Phi"], self.g(b["lam"], b["t"], b["z"]), b["coef"])
        if self.add_rp:
            u = u + b["rp"]
        if self.eps is not None:
            u = u + self.eps(b)
        return u


# ------------------------------------------------------------------------------------------------
# DON-base with input normalisation (fairness supplement, results_don_norm/)
# ------------------------------------------------------------------------------------------------
class DeepONetNorm(nn.Module):
    """DON-base with u0 standardised per (shape, IC) and the output mapped back:

        μ = Σ m u0 / Σ m,  σ = sqrt(Σ m (u0 − μ)² / Σ m)   (m = m_pc)
        û = μ + σ · DON([x, (u0 − μ)/σ], [x, log t])

    Exact for the heat equation (linear, mass-conserving): u(t; μ + σv0) = μ + σ u(t; v0).
    Coordinates keep the existing area normalisation.  Widths scale with h (h = 128 is
    DON-base): point MLP 4-h/2-h-2h, head 4h-2h-p, trunk 4-h-h-h-p.
    """

    def __init__(self, p=128, h=128):
        super().__init__()
        self.point = mlp([4, h // 2, h, 2 * h])
        self.head = mlp([4 * h, 2 * h, p])
        self.trunk = mlp([4, h, h, h, p])
        self.bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def stats(u0, m):
        w = (m / m.sum(1, keepdim=True))[:, None, :]                                  # (B,1,N)
        mu = (w * u0).sum(-1, keepdim=True)
        sd = torch.sqrt((w * (u0 - mu) ** 2).sum(-1, keepdim=True)).clamp_min(1e-8)
        return mu, sd                                                                   # (B,I,1)

    def forward(self, b):
        B, N, _ = b["x"].shape
        I, T = b["u0"].shape[1], b["t"].shape[1]
        mu, sd = self.stats(b["u0"], b["m_pc"])
        v0 = (b["u0"] - mu) / sd
        h = self.point(torch.cat([b["x"][:, None].expand(B, I, N, 3), v0[..., None]], -1))       # (B,I,N,2h)
        w = (b["m_pc"] / b["m_pc"].sum(1, keepdim=True))[:, None, :, None]
        br = self.head(torch.cat([(w * h).sum(2), h.amax(2)], -1))                               # (B,I,p)
        lt = log_t_feature(b["t"])[:, :, None, None].expand(B, T, N, 1)
        tr = self.trunk(torch.cat([b["x"][:, None].expand(B, T, N, 3), lt], -1))                 # (B,T,N,p)
        v = torch.einsum("bip,btnp->btin", br, tr) + self.bias
        return mu[:, None] + sd[:, None] * v

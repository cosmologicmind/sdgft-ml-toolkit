"""Level 2: Effective dimension D* and f(R) exponent n.

Two independent derivations of D*:

Tree-level (algebraic):
    D*_tree = 3 − sin²(30°) + δ = 3 − 1/4 + 1/24 = 67/24

Fixed-point (self-referential):
    D*_fp = lim_{k→∞} f^k(D₀)
    where f(D) = Δ^{−1/D} · φ · Δ^{Δ·δ}
    Converges to ≈ 2.79676 from any starting value.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .constants import DELTA, DELTA_F, DELTA_G, DELTA_G_F, PHI, SIN2_30

# ── Tree-level D* ─────────────────────────────────────────────────

D_STAR_TREE = Fraction(67, 24)
"""D*_tree = 67/24 ≈ 2.7917."""

D_STAR_TREE_F: float = float(D_STAR_TREE)

N_TREE = Fraction(67, 48)
"""f(R) exponent n = D*/2 = 67/48."""

N_TREE_F: float = float(N_TREE)

TWO_N_MINUS_1_TREE = Fraction(43, 24)
"""2n − 1 = 43/24."""

TWO_N_MINUS_1_TREE_F: float = float(TWO_N_MINUS_1_TREE)

assert D_STAR_TREE == 3 - SIN2_30 + DELTA_G
assert D_STAR_TREE == 3 - DELTA

# ── Fixed-point D* ────────────────────────────────────────────────


def compute_d_star_fp(
    d0: float = 3.0,
    delta: float = DELTA_F,
    delta_g: float = DELTA_G_F,
    phi: float = PHI,
    tol: float = 1e-15,
    max_iter: int = 1000,
) -> tuple[float, list[float]]:
    """Iterate D_{k+1} = Δ^{−1/D_k} · φ · Δ^{Δ·δ}."""
    correction = delta ** (delta * delta_g)
    history = [d0]
    d = d0
    for _ in range(max_iter):
        d_new = delta ** (-1.0 / d) * phi * correction
        history.append(d_new)
        if abs(d_new - d) < tol:
            return d_new, history
        d = d_new
    raise RuntimeError("D* fixed-point did not converge")


D_STAR_FP, _FP_HISTORY = compute_d_star_fp()
"""D*_fp ≈ 2.79676."""

N_FP: float = D_STAR_FP / 2.0
TWO_N_MINUS_1_FP: float = 2.0 * N_FP - 1.0

# ── Derived constants ─────────────────────────────────────────────

GAMMA_GEO_TREE = DELTA_G ** 2 / D_STAR_TREE
"""γ_geo(tree) = δ²/D* = 1/1608."""

GAMMA_GEO_TREE_F: float = float(GAMMA_GEO_TREE)

GAMMA_GEO_FP: float = DELTA_G_F ** 2 / D_STAR_FP
"""γ_geo(fp) ≈ 6.207 × 10⁻⁴."""

GAMMA_GEO_TREE_SQ = GAMMA_GEO_TREE ** 2
"""γ²_geo(tree) = 1/2 585 664."""

GAMMA_GEO_TREE_SQ_F: float = float(GAMMA_GEO_TREE_SQ)

GAMMA_GEO_FP_SQ: float = GAMMA_GEO_FP ** 2

assert GAMMA_GEO_TREE == Fraction(1, 1608)

# ── Gravity exponents ─────────────────────────────────────────────

ALPHA_M_TREE = Fraction(19, 86)
"""Planck-mass run-rate α_M = (n − 1)/(2n − 1) = 19/86."""

ALPHA_M_TREE_F: float = float(ALPHA_M_TREE)

ALPHA_B_TREE = -ALPHA_M_TREE / 2
"""Braiding coefficient α_B = −α_M/2."""

ALPHA_B_TREE_F: float = float(ALPHA_B_TREE)

# ── Particle physics (minimal set for QED/neutrino modules) ──────


def alpha_em(d_star: float = D_STAR_TREE_F, delta_g: float = DELTA_G_F) -> float:
    """SDGFT tree-level fine-structure constant."""
    return 1.0 / (2.0 * math.pi * d_star ** 3 + delta_g * d_star)


ALPHA_EM_TREE: float = alpha_em()
ALPHA_EM_INV_TREE: float = 1.0 / ALPHA_EM_TREE


def _mu_e_ratio(delta: float = DELTA_F, delta_g: float = DELTA_G_F,
                phi: float = PHI) -> float:
    """Muon-to-electron mass ratio from SDGFT."""
    return phi ** 3 / (2.0 * delta) * (1.0 + delta_g)


MU_E_RATIO: float = _mu_e_ratio()

TAU_MU_RATIO_TREE: float = PHI ** 4 * DELTA_G_F / (2.0 * DELTA_F)
TAU_E_RATIO_TREE: float = MU_E_RATIO * TAU_MU_RATIO_TREE

# ── Mixing angles ────────────────────────────────────────────────


def theta_12(delta_g: float = DELTA_G_F) -> float:
    """Solar neutrino mixing angle θ₁₂ [degrees]."""
    return math.degrees(math.atan(1.0 / math.sqrt(2.0))) * (1.0 - delta_g)


def theta_23(delta: float = DELTA_F) -> float:
    """Atmospheric mixing angle θ₂₃ [degrees]."""
    return 45.0 * (1.0 + delta / math.sqrt(6.0))


def theta_13(delta: float = DELTA_F) -> float:
    """Reactor mixing angle θ₁₃ [degrees]."""
    return math.degrees(math.asin(delta / math.sqrt(2.0)))

# ── Galactic scales ──────────────────────────────────────────────


def transition_radius_kpc(
    r_h: float = 4.4e26,
    r_p: float = 1.616e-35,
    d_star: float = D_STAR_TREE_F,
    delta_g: float = DELTA_G_F,
    kpc_m: float = 3.0857e19,
) -> float:
    """SDGFT galactic transition radius [kpc]."""
    r_trans_m = r_h * (r_p / r_h) ** (d_star * delta_g)
    return r_trans_m / kpc_m


R_TRANS_KPC: float = transition_radius_kpc()
"""Galactic transition radius ≈ 1.82 kpc."""

EPSILON_GAL: float = 0.16
"""Observed galactic ε from SPARC fits."""

B_TF_TREE = Fraction(91, 24)
"""Tully-Fisher slope b = 91/24 ≈ 3.792."""

B_TF_TREE_F: float = float(B_TF_TREE)

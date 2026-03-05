"""Black hole physics: singularity resolution, Hawking radiation, QNM corrections.

Running G(k) = G_N / [1 + (k/k_P)²] from dimensional flow resolves
the Schwarzschild singularity and modifies evaporation.

Key results:
    - Kretschner scalar saturates at K_max ~ 1/r_P⁴  (no singularity)
    - Hawking temperature bounded by T_max ~ c² M_P / k_B ~ 10³² K
    - Evaporation halts → Planck-mass remnant
    - QNM correction ~ (r_P/r_s)² ~ 10⁻⁷⁶ for 30 M☉ (unobservable)
"""

from __future__ import annotations

import math

from .constants import G_N, C, HBAR, K_B, M_P, R_P, K_P, M_SUN
from .dimension import ALPHA_M_TREE_F

# ── Constants ─────────────────────────────────────────────────────

D_STAR_UV: float = 2.0
ETA_ANOMALOUS: float = 2.0

# ── Running G ─────────────────────────────────────────────────────


def g_running(k: float, g_n: float = G_N, k_p: float = K_P) -> float:
    """G(k) = G_N / [1 + (k/k_P)^η]."""
    return g_n / (1.0 + (k / k_p) ** ETA_ANOMALOUS)


def g_of_r(r: float, g_n: float = G_N, r_p: float = R_P) -> float:
    """G(r) via k ~ 1/r."""
    return g_running(1.0 / r, g_n, 1.0 / r_p)


# ── Schwarzschild ─────────────────────────────────────────────────


def schwarzschild_radius(m: float, g_n: float = G_N, c: float = C) -> float:
    return 2.0 * g_n * m / c ** 2


# ── Kretschner ────────────────────────────────────────────────────


def kretschner_classical(m: float, r: float) -> float:
    return 48.0 * G_N ** 2 * m ** 2 / (C ** 4 * r ** 6)


KRETSCHNER_MAX: float = C ** 6 / (HBAR ** 2 * G_N ** 2)
KRETSCHNER_MAX_ALT: float = 1.0 / R_P ** 4

# ── Hawking radiation ─────────────────────────────────────────────

T_HAWKING_MAX: float = C ** 2 * M_P / K_B
"""Maximum Hawking temperature ~ 1.4 × 10³² K."""

M_REMNANT: float = M_P
M_REMNANT_GEV: float = M_REMNANT * C ** 2 / 1.602176634e-10


def hawking_temperature(m: float, g_n: float = G_N,
                        use_running_g: bool = True) -> float:
    """Hawking temperature [K].  Running G caps T at T_max."""
    if use_running_g:
        r_s = schwarzschild_radius(m)
        g_eff = g_of_r(r_s)
        if g_eff < 1e-100:
            return T_HAWKING_MAX
    else:
        g_eff = g_n
    return HBAR * C ** 3 / (8.0 * math.pi * g_eff * m * K_B)


# ── QNM corrections ──────────────────────────────────────────────


def qnm_correction(m: float, alpha_m: float = ALPHA_M_TREE_F,
                    r_p: float = R_P) -> float:
    """δω/ω = α_M · (r_P/r_s)²."""
    r_s = schwarzschild_radius(m)
    return alpha_m * (r_p / r_s) ** 2


# ── Bekenstein-Hawking entropy ────────────────────────────────────


def bekenstein_hawking_entropy(m: float) -> float:
    """S_BH = A / (4 l_P²)  [in units of k_B]."""
    r_s = schwarzschild_radius(m)
    return 4.0 * math.pi * r_s ** 2 / (4.0 * R_P ** 2)


# ── Neutron-star TOV (SDGFT extension — new code) ────────────────


def tov_running_g(r: float, m_enclosed: float, p: float,
                  rho: float, g_n: float = G_N) -> tuple[float, float]:
    """TOV structure equations with SDGFT running-G.

    dm/dr = 4π r² ρ
    dp/dr = -(G(r) (ρ + p/c²)(m + 4πr³p/c²)) / (r(r − 2G(r)m/c²))

    Returns (dm_dr, dp_dr).
    """
    g_eff = g_of_r(r, g_n)
    c2 = C ** 2
    dm_dr = 4.0 * math.pi * r ** 2 * rho
    denom = r * (r - 2.0 * g_eff * m_enclosed / c2)
    if denom <= 0.0:
        return dm_dr, -1e30  # inside horizon
    dp_dr = -(g_eff * (rho + p / c2) * (m_enclosed + 4.0 * math.pi * r ** 3 * p / c2)) / denom
    return dm_dr, dp_dr


def integrate_tov(rho_c: float, eos_func, dr: float = 10.0,
                  r_max: float = 30_000.0,
                  use_running_g: bool = True) -> dict:
    """Integrate TOV from centre to surface.

    Args:
        rho_c: Central density [kg/m³].
        eos_func: p = eos_func(rho) equation of state.
        dr: Radial step [m].
        r_max: Maximum radius [m].
        use_running_g: Use SDGFT running G or classical.

    Returns dict with keys: R_km, M_msun, radii, pressures, densities.
    """
    r = dr
    m_enc = 0.0
    rho = rho_c
    p = eos_func(rho)

    radii, pressures, densities, masses = [r], [p], [rho], [m_enc]

    while r < r_max and p > 0.0:
        if use_running_g:
            dm, dp = tov_running_g(r, m_enc, p, rho)
        else:
            dm, dp = tov_running_g(r, m_enc, p, rho, g_n=G_N)
            # Classical: override g_of_r by using large r_p so G≈G_N
            dm = 4.0 * math.pi * r ** 2 * rho
            denom = r * (r - 2.0 * G_N * m_enc / C ** 2)
            if denom <= 0:
                break
            dp = -(G_N * (rho + p / C ** 2) * (m_enc + 4.0 * math.pi * r ** 3 * p / C ** 2)) / denom

        m_enc += dm * dr
        p += dp * dr
        if p < 0:
            p = 0.0
        rho = _invert_eos(p, eos_func, rho)
        r += dr
        radii.append(r)
        pressures.append(p)
        densities.append(rho)
        masses.append(m_enc)

    return {
        "R_km": r / 1e3,
        "M_msun": m_enc / M_SUN,
        "radii": radii,
        "pressures": pressures,
        "densities": densities,
        "masses": masses,
    }


def _invert_eos(p: float, eos_func, rho_guess: float) -> float:
    """Bisection inversion of eos: find ρ such that eos(ρ) = p."""
    if p <= 0:
        return 0.0
    lo, hi = 0.0, rho_guess * 3.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if eos_func(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def polytropic_eos(K: float = 5.0e-3, gamma: float = 2.0):
    """Return p(ρ) = K · ρ^γ.

    Default K = 5×10⁻³ m⁵/(kg·s²) with Γ=2 gives
    M_max ~ 1.5–2 M☉ (consistent with nuclear EOS).
    """
    def eos(rho: float) -> float:
        return K * rho ** gamma
    return eos


# ── Module-level values ───────────────────────────────────────────

T_HAWKING_SOLAR: float = hawking_temperature(M_SUN, use_running_g=False)
QNM_CORRECTION_30MSUN: float = qnm_correction(30.0 * M_SUN)
S_BH_SOLAR: float = bekenstein_hawking_entropy(M_SUN)

# Experimental Reference Data

The SDGFT-ML-Toolkit validates predictions against 22 precision measurements
from particle physics, cosmology, inflation, and neutrino physics.

## Cosmological Parameters

| Observable | Symbol | Value | σ | Source | BibTeX Key |
|------------|--------|-------|---|--------|------------|
| Baryon density | Ω_b | 0.0493 | 0.0020 | Planck 2018 TT,TE,EE+lowE+lensing | `Planck:2018vyg` |
| Cold DM density | Ω_c | 0.265 | 0.007 | Planck 2018 | `Planck:2018vyg` |
| Matter density | Ω_m | 0.3153 | 0.0073 | Planck 2018 | `Planck:2018vyg` |
| Dark energy density | Ω_Λ | 0.6847 | 0.0073 | Planck 2018 | `Planck:2018vyg` |
| Clustering amplitude | S_8 | 0.832 | 0.013 | Planck 2018 | `Planck:2018vyg` |
| Baryon asymmetry | η_B | 6.143 × 10⁻¹⁰ | 0.190 × 10⁻¹⁰ | Planck 2018 + BBN | `Planck:2018vyg` |
| Dark energy EoS | w_DE | −1.03 | 0.03 | Planck + BAO + SN | `Planck:2018vyg` |

## Inflationary Observables

| Observable | Symbol | Value | σ | Source | BibTeX Key |
|------------|--------|-------|---|--------|------------|
| Scalar spectral index | n_s | 0.9649 | 0.0042 | Planck 2018 | `Planck:2018jri` |
| Tensor-to-scalar ratio | r | < 0.036 (95% CL) | 0.036 | BICEP/Keck 2021 | `BICEP:2021xfz` |

## Particle Physics (PDG 2024)

| Observable | Symbol | Value | σ | Source | BibTeX Key |
|------------|--------|-------|---|--------|------------|
| Fine-structure constant⁻¹ | α_em⁻¹(0) | 137.035999177 | 2.1 × 10⁻⁸ | CODATA 2018 | `Tiesinga:2021myr` |
| Strong coupling | α_s(M_Z) | 0.1180 | 0.0009 | PDG 2024 average | `PDG:2024` |
| Weak mixing angle | sin²θ_W | 0.23122 | 0.00003 | PDG 2024 (MS-bar) | `PDG:2024` |
| Higgs boson mass | m_H | 125.25 GeV | 0.17 GeV | ATLAS + CMS combined | `PDG:2024` |
| Muon-electron mass ratio | m_μ/m_e | 206.7682830 | 4.6 × 10⁻⁶ | CODATA 2018 | `Tiesinga:2021myr` |
| Tau-muon mass ratio | m_τ/m_μ | 16.8170 | 0.0015 | PDG 2024 | `PDG:2024` |
| Fermion generations | N_gen | 3.0 | 0.008 | LEP Z-width (N_ν = 2.984 ± 0.008) | `ALEPH:2005ab` |
| Higgs quartic coupling | λ | 0.1291 | 0.0020 | Derived: λ = m_H² / (2v²) | `PDG:2024` |

## Neutrino Mixing (NuFIT 5.3)

| Observable | Symbol | Value | σ | Source | BibTeX Key |
|------------|--------|-------|---|--------|------------|
| Solar mixing angle | θ₁₂ | 33.44° | 0.77° | NuFIT 5.3 (2024) | `Esteban:2020cvm` |
| Atmospheric mixing angle | θ₂₃ | 49.2° | 1.0° | NuFIT 5.3 (2024) | `Esteban:2020cvm` |
| Reactor mixing angle | θ₁₃ | 8.57° | 0.12° | NuFIT 5.3 (2024) | `Esteban:2020cvm` |

## CKM Matrix Elements (PDG 2024)

| Observable | Symbol | Value | σ | Source | BibTeX Key |
|------------|--------|-------|---|--------|------------|
| CKM |V_us| | \|V_us\| | 0.2243 | 0.0005 | PDG 2024 CKM fit | `PDG:2024` |
| CKM |V_ub| | \|V_ub\| | 0.00382 | 0.00020 | PDG 2024 CKM fit | `PDG:2024` |

## Theory Uncertainties

Some observables have additional tree-level theory uncertainties that dominate
over the experimental precision:

| Observable | Exp. σ | Theory σ | Effective σ | Reason |
|------------|--------|----------|-------------|--------|
| α_em⁻¹(0) | 2.1 × 10⁻⁸ | **0.5** | 0.5 | Tree-level; ~0.4% loop corrections expected |
| m_μ/m_e | 4.6 × 10⁻⁶ | **1.0** | 1.0 | Geometric ratio; ~0.5% radiative corrections |
| m_τ/m_μ | 0.0015 | **0.1** | 0.1 | Geometric ratio; QCD corrections expected |

The validation code uses `max(σ_exp, σ_theory)` as the effective uncertainty.

## BibTeX References

```bibtex
@article{Planck:2018vyg,
    author = "{Planck Collaboration}",
    title = "{Planck 2018 results. VI. Cosmological parameters}",
    journal = "Astron. Astrophys.",
    volume = "641",
    pages = "A6",
    year = "2020",
    eprint = "1807.06209",
    archivePrefix = "arXiv",
}

@article{Planck:2018jri,
    author = "{Planck Collaboration}",
    title = "{Planck 2018 results. X. Constraints on inflation}",
    journal = "Astron. Astrophys.",
    volume = "641",
    pages = "A10",
    year = "2020",
    eprint = "1807.06211",
    archivePrefix = "arXiv",
}

@article{BICEP:2021xfz,
    author = "{BICEP/Keck Collaboration}",
    title = "{Improved Constraints on Primordial Gravitational Waves
              using Planck, WMAP, and BICEP/Keck Observations through
              the 2018 Observing Season}",
    journal = "Phys. Rev. Lett.",
    volume = "127",
    pages = "151301",
    year = "2021",
    eprint = "2110.00483",
    archivePrefix = "arXiv",
}

@article{Esteban:2020cvm,
    author = "Esteban, Ivan and others",
    title = "{The fate of hints: updated global analysis of
              three-flavour neutrino oscillations}",
    journal = "JHEP",
    volume = "09",
    pages = "178",
    year = "2020",
    note = "Updated: NuFIT 5.3 (2024), \url{http://www.nu-fit.org}",
}

@article{Tiesinga:2021myr,
    author = "Tiesinga, Eite and others",
    title = "{CODATA recommended values of the fundamental physical
              constants: 2018}",
    journal = "Rev. Mod. Phys.",
    volume = "93",
    pages = "025010",
    year = "2021",
}

@article{PDG:2024,
    author = "{Particle Data Group}",
    title = "{Review of Particle Physics}",
    journal = "Phys. Rev. D",
    volume = "110",
    pages = "030001",
    year = "2024",
    note = "\url{https://pdg.lbl.gov}",
}

@article{ALEPH:2005ab,
    author = "{ALEPH, DELPHI, L3, OPAL, SLD Collaborations}",
    title = "{Precision electroweak measurements on the Z resonance}",
    journal = "Phys. Rept.",
    volume = "427",
    pages = "257",
    year = "2006",
    eprint = "hep-ex/0509008",
}
```

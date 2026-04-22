"""
Synthetic Binary Distillation Column Dataset Generator
=======================================================
Physically realistic dataset based on:
- Fenske-Underwood-Gilliland (FUG) shortcut method
- Murphree tray efficiency model
- Raoult's Law for VLE (benzene/toluene-like system)
- Energy balance via reboiler/condenser duty estimation

Author: Chemical Process + ML Engineering
"""

import numpy as np
import pandas as pd

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
N = 1000

# ══════════════════════════════════════════════════════════════════════════════
# 1.  INDEPENDENT VARIABLES  (physically bounded sampling)
# ══════════════════════════════════════════════════════════════════════════════

feed_composition  = np.random.uniform(0.10, 0.90, N)   # z_F  [mol frac light key]
feed_temperature  = np.random.uniform(60,   130,  N)   # T_F  [°C]
reflux_ratio      = np.random.uniform(1.2,  6.0,  N)   # R    [-]  (≥ R_min guaranteed later)
column_pressure   = np.random.uniform(80,   200,  N)   # P    [kPa]
num_trays         = np.random.randint(10,   41,   N)   # N_T  [-]  integer
feed_flow_rate    = np.random.uniform(50,   500,  N)   # F    [kmol/h]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  THERMODYNAMIC HELPERS  (Antoine / Raoult – benzene-toluene proxy)
# ══════════════════════════════════════════════════════════════════════════════

def bubble_point_temperature(z, P_kPa):
    """Estimate bubble-point T [°C] for a benzene(1)/toluene(2) mixture
    using Antoine constants and Raoult's law (iterative bubble-point)."""
    P_mmHg = P_kPa * 7.50062  # kPa → mmHg

    # Antoine constants  log10(P_sat/mmHg) = A - B/(C+T[°C])
    A1, B1, C1 = 6.90565, 1211.033, 220.790   # benzene
    A2, B2, C2 = 6.95334, 1343.943, 219.377   # toluene

    T = 80.0  # initial guess [°C]
    for _ in range(60):
        P1 = 10 ** (A1 - B1 / (C1 + T))
        P2 = 10 ** (A2 - B2 / (C2 + T))
        f  = z * P1 + (1 - z) * P2 - P_mmHg
        df = z * P1 * B1 / (C1 + T)**2 * np.log(10) + \
             (1 - z) * P2 * B2 / (C2 + T)**2 * np.log(10)
        T -= f / df
    return T

def relative_volatility(T_C, P_kPa):
    """α = P1_sat / P2_sat  at temperature T [°C]."""
    P_mmHg = P_kPa * 7.50062
    A1, B1, C1 = 6.90565, 1211.033, 220.790
    A2, B2, C2 = 6.95334, 1343.943, 219.377
    P1 = 10 ** (A1 - B1 / (C1 + T_C))
    P2 = 10 ** (A2 - B2 / (C2 + T_C))
    return P1 / P2

def y_eq(x, alpha):
    """VLE: equilibrium vapour composition from liquid x."""
    return alpha * x / (1 + (alpha - 1) * x)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  PROCESS CALCULATIONS  (row-by-row physics)
# ══════════════════════════════════════════════════════════════════════════════

distillate_purity   = np.zeros(N)
bottoms_composition = np.zeros(N)
energy_consumption  = np.zeros(N)  # [kW]
column_efficiency   = np.zeros(N)  # [%]

# Latent heat proxy [kJ/kmol] – weighted average benzene/toluene
LAMBDA = 33_000  # kJ/kmol  (~33 MJ/kmol)
Cp_liq = 140     # kJ/(kmol·°C) liquid heat capacity

for i in range(N):
    z   = feed_composition[i]
    T_F = feed_temperature[i]
    R   = reflux_ratio[i]
    P   = column_pressure[i]
    N_T = num_trays[i]
    F   = feed_flow_rate[i]

    # ── 3a. Bubble/dew point at column pressure ────────────────────────────
    T_bub = bubble_point_temperature(z, P)
    alpha = relative_volatility(T_bub, P)     # representative α at feed stage

    # ── 3b. Minimum reflux ratio  (Underwood simplified) ──────────────────
    # For a binary, saturated-liquid feed: R_min = (x_D - y_Fq) / (y_Fq - z)
    # where y_Fq is equilibrium vapour at feed composition
    y_Fq  = y_eq(z, alpha)
    # Feed condition q  (fraction liquid in feed)
    if T_F < T_bub:
        q = 1 + Cp_liq * (T_bub - T_F) / LAMBDA   # subcooled
    elif T_F < T_bub + 20:
        q = 1.0                                     # saturated liquid
    else:
        q = max(0.0, 1 - (T_F - T_bub) / 30)       # partial vaporisation

    # Target distillate / bottoms purity targets (aspirational – capped by N_T)
    x_D_ideal = min(0.995, 0.70 + 0.28 * z + 0.005 * (alpha - 1))
    x_B_ideal = max(0.005, 0.30 * z - 0.01 * (alpha - 1))

    # ── 3c. Minimum trays  (Fenske equation) ──────────────────────────────
    if x_D_ideal <= 0 or (1 - x_D_ideal) <= 0 or x_B_ideal <= 0 or (1 - x_B_ideal) <= 0:
        N_min = 5
    else:
        N_min = np.log((x_D_ideal / (1 - x_D_ideal)) *
                       ((1 - x_B_ideal) / x_B_ideal)) / np.log(max(alpha, 1.01))

    # ── 3d. Minimum reflux (Underwood) ────────────────────────────────────
    if q == 1.0:
        R_min = (x_D_ideal - y_Fq) / (y_Fq - z + 1e-9)
    else:
        R_min = max(0.5, (x_D_ideal - y_Fq) / (y_Fq - z + 1e-9) * (1 / (1 - q + 1e-9)))
    R_min = max(R_min, 0.5)

    # ── 3e. Actual reflux ratio must exceed R_min ─────────────────────────
    R_actual = max(R, R_min * 1.05)   # ensure feasibility

    # ── 3f. Gilliland correlation  (N_actual / N_min vs. X) ───────────────
    X = (R_actual - R_min) / (R_actual + 1)
    Y = 1 - np.exp((1 + 54.4 * X) / (11 + 117.2 * X) * (X - 1) / X**0.5)
    N_theoretical = N_min / (1 - Y)    # theoretical trays needed

    # ── 3g. Murphree tray efficiency ──────────────────────────────────────
    # E_mv depends on α, viscosity proxy (temperature), and reflux
    mu_proxy  = np.exp(-0.025 * (T_bub - 60))        # lower T → higher viscosity
    E_mv_base = 0.55 + 0.20 * np.log(alpha) - 0.05 * mu_proxy
    E_mv_base = np.clip(E_mv_base, 0.30, 0.85)
    # Small noise
    E_mv = E_mv_base + np.random.normal(0, 0.02)
    E_mv = np.clip(E_mv, 0.25, 0.90)

    N_actual_equivalent = N_T * E_mv   # effective theoretical stages

    # ── 3h. Achievable separation given N_T trays ─────────────────────────
    separation_ratio = N_actual_equivalent / max(N_theoretical, 1.0)
    separation_ratio = np.clip(separation_ratio, 0.30, 1.20)

    # Distillate purity degrades when column is undersized
    x_D = x_D_ideal * np.clip(separation_ratio, 0.0, 1.0)
    x_D = np.clip(x_D, 0.50, 0.999)

    # Bottoms purity worsens when column is undersized
    x_B = x_B_ideal / np.clip(separation_ratio, 0.5, 1.5)
    x_B = np.clip(x_B, 0.001, 0.45)

    # ── 3i. Material balance  →  D and B flow rates ───────────────────────
    # Overall: F = D + B ;  Component: F*z = D*x_D + B*x_B
    if abs(x_D - x_B) < 1e-6:
        x_D += 0.001
    D = F * (z - x_B) / (x_D - x_B)
    D = np.clip(D, 0.01 * F, 0.99 * F)
    B = F - D

    # ── 3j. Energy balance ────────────────────────────────────────────────
    # Vapour flow in rectifying section: V = D * (R_actual + 1)
    V = D * (R_actual + 1)          # [kmol/h]

    # Condenser duty (total condenser assumed)
    Q_cond = V * LAMBDA / 3600      # kW  (LAMBDA in kJ/kmol, /3600 → kW·h/kmol → kW)

    # Reboiler duty ≈ condenser duty + feed preheat deficit
    T_feed_actual = T_F
    T_feed_ideal  = T_bub
    Q_preheat = F * Cp_liq * max(0, T_feed_ideal - T_feed_actual) / 3600  # kW

    Q_reb = Q_cond + Q_preheat      # kW  (energy penalty for cold feed)
    Q_total = Q_cond + Q_reb        # total column duty [kW]

    # Add small Gaussian noise (instrument/model uncertainty ±3 %)
    Q_total *= (1 + np.random.normal(0, 0.03))

    # ── 3k. Column efficiency  (overall) ──────────────────────────────────
    eff = E_mv * 100  # [%]
    eff = np.clip(eff + np.random.normal(0, 1.0), 25, 90)

    # ── Store results ──────────────────────────────────────────────────────
    distillate_purity[i]   = round(x_D,  4)
    bottoms_composition[i] = round(x_B,  4)
    energy_consumption[i]  = round(max(Q_total, 10.0), 2)
    column_efficiency[i]   = round(eff,  2)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  ASSEMBLE DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame({
    # Inputs
    "feed_composition_molfrac": np.round(feed_composition, 4),
    "feed_temperature_C":       np.round(feed_temperature, 2),
    "reflux_ratio":             np.round(reflux_ratio,     3),
    "column_pressure_kPa":      np.round(column_pressure,  1),
    "num_trays":                num_trays.astype(int),
    "feed_flow_rate_kmolph":    np.round(feed_flow_rate,   2),
    # Outputs
    "distillate_purity_molfrac":   distillate_purity,
    "bottoms_composition_molfrac": bottoms_composition,
    "energy_consumption_kW":       energy_consumption,
    "column_efficiency_pct":       column_efficiency,
})

# ══════════════════════════════════════════════════════════════════════════════
# 5.  QUICK SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

assert (df["distillate_purity_molfrac"]   > df["feed_composition_molfrac"]).mean() > 0.85, \
    "Distillate should be richer than feed for most rows"
assert (df["bottoms_composition_molfrac"] < df["feed_composition_molfrac"]).mean() > 0.85, \
    "Bottoms should be leaner than feed for most rows"
assert df["energy_consumption_kW"].min() > 0, "Energy must be positive"
assert df["column_efficiency_pct"].between(0, 100).all(), "Efficiency must be 0–100 %"

print("=" * 60)
print("Distillation Column Synthetic Dataset  –  Summary")
print("=" * 60)
print(df.describe().T.to_string())
print(f"\nShape : {df.shape}")
print(f"NaNs  : {df.isna().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  SAVE
# ══════════════════════════════════════════════════════════════════════════════

csv_path = "distillation_synthetic_dataset.csv"
df.to_csv(csv_path, index=False)
print(f"\nDataset saved → {csv_path}")

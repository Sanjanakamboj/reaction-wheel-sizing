#!/usr/bin/env python3
"""
Milestone 3 analysis script.

Prints a concise disturbance/momentum-accumulation/saturation-time report
and generates:

    results/fig1_disturbance_components.png
    results/fig2_wheel_momentum_histories.png
    results/fig3_momentum_utilization_3v4.png
    results/fig4_failure_saturation_time.png
    results/fig5_saturation_vs_disturbance_magnitude.png
    results/momentum_budget_table.md
    results/configuration_comparison_table.md

Run with:  python scripts/analyze_momentum_accumulation.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.wheel import representative_wheel
from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel
from reaction_wheel.disturbances import (
    leo_orbit_reference, srp_torque, aero_torque, gravity_gradient_torque,
    magnetic_dipole_torque, dipole_field_body, constant_disturbance,
    CompositeDisturbance, SOLAR_PRESSURE_1AU, MU_EARTH,
)
from reaction_wheel.momentum import (
    integrate_wheel_momentum, mean_wheel_torque, momentum_per_orbit,
    first_threshold_crossing, momentum_utilization,
    analytical_constant_torque_saturation_time, secular_momentum_approximation,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hr(char="-", n=90):
    print(char * n)


def build_environment(sc, orb):
    """Baseline representative disturbance environment (illustrative
    engineering assumptions, not a mission-specific environmental model
    -- see docs/momentum_accumulation_methodology.md for every parameter's
    rationale)."""
    I = sc.inertia_tensor
    mu, r, n = MU_EARTH, orb.radius_m, orb.mean_motion

    tau_srp_vec = srp_torque(P_srp=SOLAR_PRESSURE_1AU, C_R=1.3, A=2.0,
                              r_cp=[0.0, 0.0, 0.05], u_sun=[1.0, 0.0, 0.0])
    tau_aero_vec = aero_torque(rho=5e-13, v=orb.orbital_speed_mps, C_D=2.2, A=2.0,
                                r_cp=[0.0, 0.03, 0.0], u_drag=[1.0, 0.0, 0.0])
    m_res = np.array([0.05, 0.02, -0.03])  # A*m^2, representative residual dipole

    def gg_component(t):
        r_hat_b = np.array([np.cos(n * t), np.sin(n * t), 0.0])
        return gravity_gradient_torque(mu, r, r_hat_b, I)

    def mag_component(t):
        B = dipole_field_body(t, orb)
        return magnetic_dipole_torque(m_res, B)

    env = CompositeDisturbance()
    env.add("srp", constant_disturbance(tau_srp_vec))
    env.add("aero", constant_disturbance(tau_aero_vec))
    env.add("gravity_gradient", gg_component)
    env.add("magnetic", mag_component)
    return env, dict(tau_srp_vec=tau_srp_vec, tau_aero_vec=tau_aero_vec, m_res=m_res)


def main():
    sc = representative_spacecraft()
    wheel = representative_wheel()
    orb = leo_orbit_reference(500.0)
    T_orb = orb.period_s

    f_H = 0.8  # operational momentum threshold fraction (headroom reserved)
    H_max = wheel.H_max
    H_threshold = f_H * H_max

    env, params = build_environment(sc, orb)

    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()

    hr("=")
    print("MILESTONE 3 ANALYSIS REPORT — Disturbance Momentum Accumulation & Saturation Time")
    hr("=")

    print("\nOrbital timing reference (LEO, 500 km circular, timing only -- not a propagator):")
    print(f"  radius = {orb.radius_m/1000:.1f} km, period T_orb = {T_orb:.1f} s = {T_orb/60:.2f} min, "
          f"v = {orb.orbital_speed_mps:.1f} m/s, inclination = {np.rad2deg(orb.inclination_rad):.1f} deg")

    print("\nRepresentative wheel capability (from M1):")
    print(f"  J_w = {wheel.J_w} kg*m^2, Omega_max = {wheel.Omega_max_rpm:.0f} rpm, "
          f"H_max = J_w*Omega_max = {H_max:.4f} N*m*s")
    print(f"  Operational momentum threshold: H_threshold = f_H * H_max, f_H = {f_H} "
          f"-> H_threshold = {H_threshold:.4f} N*m*s per wheel")

    # -----------------------------------------------------------------
    # Disturbance assumptions and per-component mean/peak
    # -----------------------------------------------------------------
    hr()
    print("DISTURBANCE ENVIRONMENT — assumptions and per-component statistics (over one orbit)")
    hr()
    print("  SRP:  P_srp=4.56e-6 N/m^2 (standard, near-1AU), C_R=1.3, A=2.0 m^2, "
          "r_cp=[0,0,0.05] m, u_sun=[1,0,0] (fixed worst-case direction)")
    print(f"    -> constant body torque = {np.round(params['tau_srp_vec'], 8)} N*m")
    print("  Aero: rho=5e-13 kg/m^3 (representative 500km solar-avg), v=orbital speed, "
          "C_D=2.2, A=2.0 m^2, r_cp=[0,0.03,0] m, u_drag=[1,0,0]")
    print(f"    -> constant body torque = {np.round(params['tau_aero_vec'], 8)} N*m")
    print("  Gravity-gradient: mu, r from orbit; r_hat_b(t) sweeps the body x-y plane once "
          "per orbit at the orbital rate (illustrative inertially-pointed-attitude model)")
    print(f"  Magnetic: B0=3e-5 T representative LEO field, inclination-tilted sweep once per "
          f"orbit; m_res = {params['m_res']} A*m^2 (representative residual dipole)")

    print(f"\n  {'Component':<18}{'mean |tau|':>14}{'peak |tau|':>14}{'mean vector [N*m]':>40}")
    means = {}
    peaks = {}
    for name in env.names:
        mean_vec = env.mean_component(name, T_orb, n=4000)
        peak_mag = env.peak_component_magnitude(name, T_orb, n=4000)
        means[name] = mean_vec
        peaks[name] = peak_mag
        print(f"  {name:<18}{np.linalg.norm(mean_vec):>14.3e}{peak_mag:>14.3e}"
              f"{np.array2string(mean_vec, precision=3):>40}")

    total_mean = env.mean_total(T_orb, n=4000)
    total_peak = env.peak_total_magnitude(T_orb, n=4000)
    print(f"  {'TOTAL':<18}{np.linalg.norm(total_mean):>14.3e}{total_peak:>14.3e}"
          f"{np.array2string(total_mean, precision=3):>40}")

    dominant_secular = max(means, key=lambda k: np.linalg.norm(means[k]))
    dominant_peak = max(peaks, key=lambda k: peaks[k])
    print(f"\n  Dominant SECULAR (mean-momentum-accumulation) driver: {dominant_secular} "
          f"(|mean| = {np.linalg.norm(means[dominant_secular]):.3e} N*m)")
    print(f"  Dominant PEAK-TORQUE driver: {dominant_peak} (peak = {peaks[dominant_peak]:.3e} N*m)")
    if dominant_secular != dominant_peak:
        print("  NOTE: these are DIFFERENT components -- largest instantaneous torque does "
              "NOT determine the dominant long-term momentum-accumulation driver.")

    # -----------------------------------------------------------------
    # Numerical integration over a detailed multi-orbit horizon (nominal 4-wheel)
    # -----------------------------------------------------------------
    hr()
    print("WHEEL MOMENTUM INTEGRATION (detailed horizon, nominal 4-wheel tetrahedral)")
    hr()
    n_orbits_detail = 20
    t_detail = np.linspace(0.0, n_orbits_detail * T_orb, n_orbits_detail * 60)
    # A longer horizon purely for Figure 2, so the drift toward the
    # operational threshold is actually visible (20 orbits is too short a
    # fraction of the ~1500-orbit saturation time to show any visible trend).
    n_orbits_plot = 400
    t_plot = np.linspace(0.0, n_orbits_plot * T_orb, n_orbits_plot * 30)
    hist4_detail = integrate_wheel_momentum(g4, env, t_detail)
    hist3_detail = integrate_wheel_momentum(g3, env, t_detail)

    print(f"  Integrated over {n_orbits_detail} orbits ({t_detail[-1]:.0f} s), "
          f"{len(t_detail)} samples")
    print(f"  4-wheel final |h_w| per wheel: {np.round(np.abs(hist4_detail.h_w[-1, :]), 5)} N*m*s")
    print(f"  3-wheel final |h_w| per wheel: {np.round(np.abs(hist3_detail.h_w[-1, :]), 5)} N*m*s")

    # -----------------------------------------------------------------
    # Per-orbit accumulation and mean-torque secular approximation
    # -----------------------------------------------------------------
    hr()
    print("PER-ORBIT MOMENTUM ACCUMULATION & MEAN-TORQUE APPROXIMATION")
    hr()
    delta_h4_orbit = momentum_per_orbit(g4, env, T_orb, n=4000)
    delta_h3_orbit = momentum_per_orbit(g3, env, T_orb, n=4000)
    print(f"  4-wheel Delta h_w per orbit: {np.round(delta_h4_orbit, 6)} N*m*s "
          f"(worst wheel {g4.labels[int(np.argmax(np.abs(delta_h4_orbit)))]})")
    print(f"  3-wheel Delta h_w per orbit: {np.round(delta_h3_orbit, 6)} N*m*s "
          f"(worst wheel {g3.labels[int(np.argmax(np.abs(delta_h3_orbit)))]})")

    tau_w_mean4 = mean_wheel_torque(g4, env, T_orb, n=4000)
    approx4 = secular_momentum_approximation(np.zeros(4), tau_w_mean4, t_detail)
    approx_residual4 = np.max(np.abs(approx4 - hist4_detail.h_w))
    print(f"  Mean-torque secular approximation vs full {n_orbits_detail}-orbit numerical "
          f"integration, max residual (4-wheel): {approx_residual4:.3e} N*m*s")
    print(f"  (Approximation quality over {n_orbits_detail} orbits: "
          f"{'GOOD -- residual << periodic-term amplitude' if approx_residual4 < 0.05*H_threshold else 'periodic terms are significant relative to threshold'})")

    # -----------------------------------------------------------------
    # Threshold-crossing time using the mean-torque estimate (long horizon)
    # -----------------------------------------------------------------
    hr()
    print("THRESHOLD-CROSSING TIME (analytical, mean-torque estimate; nominal geometries)")
    hr()
    results_nominal = {}
    for name, g, tau_w_mean, delta_orbit in (
        ("3-wheel orthogonal", g3, mean_wheel_torque(g3, env, T_orb, n=4000), delta_h3_orbit),
        ("4-wheel tetrahedral", g4, tau_w_mean4, delta_h4_orbit),
    ):
        t_sats = [analytical_constant_torque_saturation_time(0.0, tw, H_threshold) for tw in tau_w_mean]
        finite_t_sats = [(i, ts) for i, ts in enumerate(t_sats) if ts is not None]
        if finite_t_sats:
            worst_i, t_sat = min(finite_t_sats, key=lambda pair: pair[1])
            orbits_to_threshold = t_sat / T_orb
        else:
            worst_i, t_sat, orbits_to_threshold = None, None, None
        results_nominal[name] = dict(t_sat=t_sat, worst_wheel=g.labels[worst_i] if worst_i is not None else None,
                                      orbits=orbits_to_threshold, tau_w_mean=tau_w_mean)
        print(f"  {name}: limiting wheel {g.labels[worst_i] if worst_i is not None else 'n/a'}, "
              f"t_sat = {t_sat:.3e} s = {orbits_to_threshold:.1f} orbits ({t_sat/86400:.2f} days)"
              if t_sat is not None else f"  {name}: no wheel reaches threshold under mean-torque estimate")

    # -----------------------------------------------------------------
    # Verify with direct numerical integration out to the estimated t_sat
    # -----------------------------------------------------------------
    hr()
    print("NUMERICAL VERIFICATION OF THRESHOLD-CROSSING TIME (4-wheel, direct integration)")
    hr()
    t_sat_est = results_nominal["4-wheel tetrahedral"]["t_sat"]
    t_horizon = 1.3 * t_sat_est
    n_samples = min(30000, max(2000, int(t_horizon / T_orb) * 40))
    t_long = np.linspace(0.0, t_horizon, n_samples)
    hist4_long = integrate_wheel_momentum(g4, env, t_long)
    crossing4 = first_threshold_crossing(hist4_long, H_threshold, labels=g4.labels)
    print(f"  Numerical horizon: {t_horizon:.3e} s ({t_horizon/T_orb:.1f} orbits), {n_samples} samples")
    if crossing4.reached:
        print(f"  Numerical crossing: t = {crossing4.time:.3e} s ({crossing4.time/T_orb:.1f} orbits), "
              f"wheel {crossing4.wheel_label}, h_w = {crossing4.momentum_at_threshold:.4f} N*m*s")
        print(f"  Analytical mean-torque estimate: t = {t_sat_est:.3e} s "
              f"({t_sat_est/T_orb:.1f} orbits) -- "
              f"relative difference: {abs(crossing4.time - t_sat_est)/t_sat_est*100:.2f}%")
    else:
        print("  Threshold not reached within the numerical horizon.")

    # -----------------------------------------------------------------
    # 3-wheel vs 4-wheel comparison
    # -----------------------------------------------------------------
    hr()
    print("3-WHEEL vs 4-WHEEL MOMENTUM-ACCUMULATION COMPARISON")
    hr()
    t_sat_3 = results_nominal["3-wheel orthogonal"]["t_sat"]
    print(f"  3-wheel orthogonal:  t_sat = {t_sat_3:.3e} s ({t_sat_3/T_orb:.1f} orbits), "
          f"limiting wheel {results_nominal['3-wheel orthogonal']['worst_wheel']}")
    print(f"  4-wheel tetrahedral: t_sat = {t_sat_est:.3e} s ({t_sat_est/T_orb:.1f} orbits), "
          f"limiting wheel {results_nominal['4-wheel tetrahedral']['worst_wheel']}")
    ratio_34 = t_sat_est / t_sat_3
    verdict = "LONGER" if ratio_34 > 1 else "SHORTER"
    print(f"  4-wheel/3-wheel saturation-time ratio: {ratio_34:.3f} "
          f"-> the 4-wheel geometry gives a {verdict} time to threshold for this disturbance "
          "environment (redistribution across 4 wheels does not guarantee a longer lifetime; "
          "quantified here, not assumed).")

    # -----------------------------------------------------------------
    # Nominal vs single-wheel-failure comparison (4-wheel)
    # -----------------------------------------------------------------
    hr()
    print("NOMINAL vs SINGLE-WHEEL-FAILURE MOMENTUM-ACCUMULATION COMPARISON (4-wheel)")
    hr()
    failure_t_sats = {}
    for idx in range(4):
        g_failed = g4.remove_wheel(idx)
        tau_w_mean_f = mean_wheel_torque(g_failed, env, T_orb, n=4000)
        t_sats_f = [analytical_constant_torque_saturation_time(0.0, tw, H_threshold) for tw in tau_w_mean_f]
        finite = [(i, ts) for i, ts in enumerate(t_sats_f) if ts is not None]
        worst_i, t_sat_f = min(finite, key=lambda pair: pair[1])
        failure_t_sats[idx] = t_sat_f
        print(f"  Wheel {g4.labels[idx]} failed -> limiting wheel {g_failed.labels[worst_i]}, "
              f"t_sat = {t_sat_f:.3e} s ({t_sat_f/T_orb:.1f} orbits)")

    mean_failed_t_sat = np.mean(list(failure_t_sats.values()))
    degradation_pct = (1.0 - mean_failed_t_sat / t_sat_est) * 100.0
    print(f"\n  Nominal 4-wheel t_sat: {t_sat_est:.3e} s ({t_sat_est/T_orb:.1f} orbits)")
    print(f"  Mean single-failure t_sat: {mean_failed_t_sat:.3e} s ({mean_failed_t_sat/T_orb:.1f} orbits)")
    print(f"  Reduction in time-to-unloading after one wheel fails: {degradation_pct:.1f}%")

    # -----------------------------------------------------------------
    # Disturbance sensitivity sweep (secular SRP+aero and magnetic dipole)
    # -----------------------------------------------------------------
    hr()
    print("DISTURBANCE SENSITIVITY SWEEP")
    hr()
    scale_factors = [0.5, 1.0, 2.0]
    sensitivity_results = []
    for scale in scale_factors:
        tau_srp_scaled = params["tau_srp_vec"] * scale
        tau_aero_scaled = params["tau_aero_vec"] * scale
        m_res_scaled = params["m_res"] * scale

        env_scaled = CompositeDisturbance()
        env_scaled.add("srp", constant_disturbance(tau_srp_scaled))
        env_scaled.add("aero", constant_disturbance(tau_aero_scaled))

        def gg_component(t):
            r_hat_b = np.array([np.cos(orb.mean_motion * t), np.sin(orb.mean_motion * t), 0.0])
            return gravity_gradient_torque(MU_EARTH, orb.radius_m, r_hat_b, sc.inertia_tensor)
        env_scaled.add("gravity_gradient", gg_component)

        def mag_component(t, m_res_scaled=m_res_scaled):
            B = dipole_field_body(t, orb)
            return magnetic_dipole_torque(m_res_scaled, B)
        env_scaled.add("magnetic", mag_component)

        tau_w_mean_s = mean_wheel_torque(g4, env_scaled, T_orb, n=4000)
        t_sats_s = [analytical_constant_torque_saturation_time(0.0, tw, H_threshold) for tw in tau_w_mean_s]
        finite_s = [(i, ts) for i, ts in enumerate(t_sats_s) if ts is not None]
        worst_i_s, t_sat_s = min(finite_s, key=lambda pair: pair[1])
        sensitivity_results.append((scale, t_sat_s))
        print(f"  scale = {scale:.1f}x: t_sat = {t_sat_s:.3e} s ({t_sat_s/T_orb:.1f} orbits), "
              f"limiting wheel {g4.labels[worst_i_s]}")

    base_scale, base_t_sat = sensitivity_results[1]
    print("\n  Verifying tau_d -> k*tau_d  =>  t_sat -> t_sat/k (constant-disturbance scaling law):")
    for scale, t_sat_s in sensitivity_results:
        expected_pure_inverse = base_t_sat / (scale / base_scale)
        rel_dev = abs(t_sat_s - expected_pure_inverse) / expected_pure_inverse * 100.0
        print(f"    scale={scale}x: actual t_sat={t_sat_s:.3e}s vs. pure-inverse-scaling "
              f"prediction={expected_pure_inverse:.3e}s (relative deviation {rel_dev:.2f}%)")
    print("  Deviation is near-zero here because every non-negligible secular contributor "
          "(SRP, aero, magnetic mean) was scaled together and gravity-gradient's own orbit-mean "
          "is negligible for this geometry -- the inverse law holds almost exactly. A sweep that "
          "scaled only ONE secular source while leaving another fixed would show the expected "
          "deviation from pure 1/scale instead.")

    # -----------------------------------------------------------------
    # Initial wheel-speed bias study
    # -----------------------------------------------------------------
    hr()
    print("INITIAL WHEEL-MOMENTUM BIAS STUDY (4-wheel, mean-torque estimate)")
    hr()
    bias_cases = {
        "zero bias": np.zeros(4),
        "moderate positive bias (+3 N*m*s all wheels)": np.full(4, 3.0),
        "balanced signed bias (+3,-3,+3,-3)": np.array([3.0, -3.0, 3.0, -3.0]),
    }
    for label, h0 in bias_cases.items():
        t_sats_b = [analytical_constant_torque_saturation_time(h0[i], tau_w_mean4[i], H_threshold)
                    for i in range(4)]
        finite_b = [(i, ts) for i, ts in enumerate(t_sats_b) if ts is not None]
        if finite_b:
            worst_i_b, t_sat_b = min(finite_b, key=lambda pair: pair[1])
            print(f"  {label}: t_sat = {t_sat_b:.3e} s ({t_sat_b/T_orb:.1f} orbits), "
                  f"limiting wheel {g4.labels[worst_i_b]}")
        else:
            print(f"  {label}: no wheel reaches threshold under mean-torque estimate")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig1 = os.path.join(RESULTS_DIR, "fig1_disturbance_components.png")
    fig2 = os.path.join(RESULTS_DIR, "fig2_wheel_momentum_histories.png")
    fig3 = os.path.join(RESULTS_DIR, "fig3_momentum_utilization_3v4.png")
    fig4 = os.path.join(RESULTS_DIR, "fig4_failure_saturation_time.png")
    fig5 = os.path.join(RESULTS_DIR, "fig5_saturation_vs_disturbance_magnitude.png")

    make_figure1(env, T_orb, fig1)
    hist4_plot = integrate_wheel_momentum(g4, env, t_plot)
    make_figure2(hist4_plot, g4, H_threshold, T_orb, n_orbits_plot, fig2)
    make_figure3(g3, g4, env, T_orb, H_threshold, fig3)
    make_figure4(t_sat_est, failure_t_sats, g4, T_orb, fig4)
    make_figure5(sensitivity_results, T_orb, fig5)

    print(f"\nFigures written to: {RESULTS_DIR}")
    for p in (fig1, fig2, fig3, fig4, fig5):
        print(f"  {os.path.relpath(p)}")

    # -----------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------
    budget_path = os.path.join(RESULTS_DIR, "momentum_budget_table.md")
    write_momentum_budget_table(env, means, peaks, T_orb, g4, H_threshold, budget_path)
    print(f"\nMomentum-budget table written to: {os.path.relpath(budget_path)}")

    config_path = os.path.join(RESULTS_DIR, "configuration_comparison_table.md")
    write_configuration_table(results_nominal, failure_t_sats, g3, g4, T_orb, H_threshold, config_path)
    print(f"Configuration comparison table written to: {os.path.relpath(config_path)}")

    hr("=")
    print("END OF M3 ANALYSIS REPORT")
    hr("=")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure1(env, T_orb, path):
    n_orbits = 3
    t = np.linspace(0, n_orbits * T_orb, 2000)
    fig, axs = plt.subplots(len(env.names) + 1, 1, figsize=(9, 12), sharex=True)
    for ax, name in zip(axs[:-1], env.names):
        vals = np.array([env.component(name, tk) for tk in t])
        for axis_i, axis_lbl, color in zip(range(3), ("x", "y", "z"), ("tab:red", "tab:green", "tab:blue")):
            ax.plot(t / T_orb, vals[:, axis_i], label=f"{axis_lbl}", color=color, linewidth=1.0)
        ax.set_ylabel("N*m")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    total = np.array([env.total(tk) for tk in t])
    axs[-1].plot(t / T_orb, np.linalg.norm(total, axis=1), color="black")
    axs[-1].set_ylabel("|tau_total| [N*m]")
    axs[-1].set_xlabel("Orbits")
    axs[-1].set_title("Total disturbance torque magnitude")
    axs[-1].grid(True, alpha=0.3)

    fig.suptitle("Figure 1 — Disturbance Torque Components vs Time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure2(history, geometry, H_threshold, T_orb, n_orbits, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for i in range(geometry.n_wheels):
        ax.plot(history.t / T_orb, history.h_w[:, i], label=f"Wheel {geometry.labels[i]}", color=colors[i])
    ax.axhline(H_threshold, color="black", linestyle="--", linewidth=1.2, label=r"$+H_{\rm threshold}$")
    ax.axhline(-H_threshold, color="black", linestyle="--", linewidth=1.2, label=r"$-H_{\rm threshold}$")
    ax.set_xlabel("Orbits")
    ax.set_ylabel(r"Wheel momentum $h_w$ [N·m·s]")
    ax.set_title(f"Figure 2 — Wheel Momentum Histories, Nominal 4-Wheel ({n_orbits} orbits)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure3(g3, g4, env, T_orb, H_threshold, path):
    n_orbits = 200
    t = np.linspace(0, n_orbits * T_orb, 6000)
    hist3 = integrate_wheel_momentum(g3, env, t)
    hist4 = integrate_wheel_momentum(g4, env, t)
    util3 = momentum_utilization(hist3, H_threshold)
    util4 = momentum_utilization(hist4, H_threshold)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t / T_orb, util3, label="3-wheel orthogonal", color="tab:blue")
    ax.plot(t / T_orb, util4, label="4-wheel tetrahedral", color="tab:orange")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label=r"threshold ($\rho_H=1$)")
    ax.set_xlabel("Orbits")
    ax.set_ylabel(r"$\rho_H(t) = \max_i |h_{w,i}(t)|/H_{\rm threshold}$")
    ax.set_title("Figure 3 — Momentum Utilization: 3-Wheel vs 4-Wheel")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure4(t_sat_nominal, failure_t_sats, g4, T_orb, path):
    labels = ["Nominal\n(4 wheels)"] + [f"{g4.labels[i]}\nfailed" for i in range(4)]
    values_orbits = [t_sat_nominal / T_orb] + [failure_t_sats[i] / T_orb for i in range(4)]
    colors = ["black", "tab:red", "tab:green", "tab:blue", "tab:purple"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values_orbits, color=colors, alpha=0.85)
    ax.set_ylabel("Orbits to threshold crossing")
    ax.set_title("Figure 4 — Nominal vs Single-Wheel-Failure Saturation Time")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, values_orbits):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure5(sensitivity_results, T_orb, path):
    scales = np.array([s for s, _ in sensitivity_results])
    t_sats = np.array([t for _, t in sensitivity_results])
    base_scale, base_t_sat = sensitivity_results[1]
    pure_inverse = base_t_sat * base_scale / scales

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(scales, t_sats / T_orb, "o-", color="tab:blue", label="Actual (mean-torque estimate)")
    ax.plot(scales, pure_inverse / T_orb, "--", color="tab:gray", label=r"Pure $1/{\rm scale}$ (secular-only)")
    ax.set_xlabel("Secular disturbance (SRP + aero + magnetic) scale factor")
    ax.set_ylabel("Orbits to threshold")
    ax.set_yscale("log")
    ax.set_title("Figure 5 — Saturation Time vs Secular Disturbance Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_momentum_budget_table(env, means, peaks, T_orb, g4, H_threshold, path):
    lines = ["# Milestone 3 — Momentum Budget Table\n"]
    lines.append(
        f"Operational momentum threshold H_threshold = {H_threshold:.4f} N*m*s per wheel "
        f"(4-wheel tetrahedral geometry shown for wheel momentum increment).\n"
    )
    lines.append(
        "| Disturbance component | Mean body torque [N·m] | Peak body torque [N·m] | "
        "Worst wheel Δh/orbit [N·m·s] | Orbits to threshold (component alone) |"
    )
    lines.append("|---|---|---|---|---|")
    for name in env.names:
        delta_h_orbit = momentum_per_orbit(g4, lambda t, name=name: env.component(name, t), T_orb, n=4000)
        worst_i = int(np.argmax(np.abs(delta_h_orbit)))
        tau_w_mean = mean_wheel_torque(g4, lambda t, name=name: env.component(name, t), T_orb, n=4000)
        t_sat = analytical_constant_torque_saturation_time(0.0, tau_w_mean[worst_i], H_threshold)
        orbits_str = f"{t_sat/T_orb:.1f}" if t_sat is not None else "not reached (bounded/zero-mean)"
        lines.append(
            f"| {name} | {np.linalg.norm(means[name]):.3e} | {peaks[name]:.3e} | "
            f"{delta_h_orbit[worst_i]:.3e} ({g4.labels[worst_i]}) | {orbits_str} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_configuration_table(results_nominal, failure_t_sats, g3, g4, T_orb, H_threshold, path):
    lines = ["# Milestone 3 — Configuration Comparison Table\n"]
    lines.append(f"Combined baseline disturbance environment; H_threshold = {H_threshold:.4f} N*m*s.\n")
    lines.append("| Configuration | Limiting wheel | t_sat [s] | Orbits to threshold | Days to threshold |")
    lines.append("|---|---|---|---|---|")
    for name, r in results_nominal.items():
        lines.append(f"| {name} (nominal) | {r['worst_wheel']} | {r['t_sat']:.3e} | "
                      f"{r['orbits']:.1f} | {r['t_sat']/86400:.2f} |")
    for idx in range(4):
        t_sat = failure_t_sats[idx]
        lines.append(f"| 4-wheel, {g4.labels[idx]} failed | (surviving 3 wheels) | {t_sat:.3e} | "
                      f"{t_sat/T_orb:.1f} | {t_sat/86400:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

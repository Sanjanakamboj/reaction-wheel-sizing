#!/usr/bin/env python3
"""
Milestone 4 analysis script.

Prints a concise magnetorquer/desaturation/schedule report and generates:

    results/fig1_single_dump_event.png
    results/fig2_magnetic_unloading_geometry.png
    results/fig3_long_duration_cycles.png
    results/fig4_dump_duration_vs_dipole_capability.png
    results/fig5_threshold_band_trade.png
    results/desaturation_table.md
    results/desaturation_schedule.csv

Run with:  python scripts/analyze_desaturation.py
"""

import os
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_momentum_accumulation import build_environment  # reuse M3's environment, not duplicated

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.wheel import representative_wheel
from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel, per_wheel_torque_utilization
from reaction_wheel.disturbances import leo_orbit_reference, dipole_field_body
from reaction_wheel.momentum import mean_wheel_torque
from reaction_wheel.desaturation import (
    Magnetorquer, representative_magnetorquer, HysteresisThresholds,
    simulate_dump, analytical_repeat_interval, null_space_redistribute,
    simulate_mission_schedule, schedule_statistics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hr(char="-", n=92):
    print(char * n)


def natural_h0(geometry, tau_w_mean, H_on):
    """Scale the natural secular-accumulation direction so the worst wheel
    sits exactly at H_on -- a representative 'just reached threshold'
    initial condition, derived from the environment rather than picked
    arbitrarily."""
    peak = np.max(np.abs(tau_w_mean))
    return tau_w_mean / peak * H_on


def main():
    sc = representative_spacecraft()
    wheel = representative_wheel()
    orb = leo_orbit_reference(500.0)
    T_orb = orb.period_s
    env, _ = build_environment(sc, orb)
    B_field = lambda t: dipole_field_body(t, orb)

    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()
    mt = representative_magnetorquer()

    H_max = wheel.H_max
    H_on = 0.8 * H_max
    H_off = 0.4 * H_max
    thresholds = HysteresisThresholds(H_on=H_on, H_off=H_off)
    k_H = 0.0005  # 1/s, representative proportional momentum-feedback gain

    hr("=")
    print("MILESTONE 4 ANALYSIS REPORT — Momentum Dumping, Desaturation Logic & Schedule")
    hr("=")

    print(f"\nMagnetorquer: {mt.name}, m_max = {mt.m_max} A*m^2")
    print(f"Representative LEO field magnitude ~3e-5 T (M3 model); "
          f"max torque authority = m_max*|B| ~ {mt.max_torque([0, 0, 3e-5]):.3e} N*m")
    print(f"\nOperational thresholds: H_on = 0.8*H_max = {H_on:.4f} N*m*s, "
          f"H_off = 0.4*H_max = {H_off:.4f} N*m*s (representative headroom/hysteresis choice)")
    print(f"Desaturation gain k_H = {k_H} 1/s (proportional momentum feedback; "
          f"1/k_H = {1/k_H:.0f} s ideal, unsaturated, full-effectiveness time constant)")

    # -----------------------------------------------------------------
    # Baseline single-dump event (nominal 4-wheel)
    # -----------------------------------------------------------------
    hr()
    print("BASELINE SINGLE DESATURATION EVENT (nominal 4-wheel tetrahedral)")
    hr()
    tau_w_mean4 = mean_wheel_torque(g4, env, T_orb, n=4000)
    h0_4 = natural_h0(g4, tau_w_mean4, H_on)
    print(f"  Initial wheel momentum (natural secular direction, worst wheel at H_on): "
          f"{np.round(h0_4, 4)} N*m*s")

    dt_dump, t_max_dump = 20.0, 15 * T_orb
    dump4 = simulate_dump(g4, h0_4, env, B_field, mt, k_H, H_off, dt_dump, t_max_dump)
    ideal_exp_time = np.log(H_on / H_off) / k_H

    print(f"  Reached H_off: {dump4.reached_off}")
    print(f"  Dump duration: {dump4.duration:.1f} s = {dump4.duration/T_orb:.2f} orbits "
          f"= {dump4.duration/3600:.2f} hours")
    print(f"  Ideal (unsaturated, full-effectiveness) exponential estimate: "
          f"{ideal_exp_time:.1f} s = {ideal_exp_time/T_orb:.2f} orbits "
          f"(actual is longer due to field-geometry effectiveness < 1 and dipole saturation)")
    print(f"  Final wheel momentum: {np.round(dump4.h_w_final, 4)} N*m*s")
    dipole_norms = np.linalg.norm(dump4.m_cmd, axis=1)
    print(f"  Max |m_cmd| = {np.max(dipole_norms):.3f} A*m^2 "
          f"(m_max = {mt.m_max}, utilization = {np.max(dipole_norms)/mt.m_max:.3f})")
    print(f"  Mean |m_cmd| = {np.mean(dipole_norms):.3f} A*m^2")
    tau_ach_norms = np.linalg.norm(dump4.tau_achieved, axis=1)
    print(f"  Peak achieved unloading torque: {np.max(tau_ach_norms):.3e} N*m")
    print(f"  Mean field-geometry effectiveness: {np.mean(dump4.effectiveness):.3f} "
          f"(min {np.min(dump4.effectiveness):.3f}, max {np.max(dump4.effectiveness):.3f})")

    rho_tau_dump = per_wheel_torque_utilization(dump4.tau_w, wheel.tau_max)
    print(f"  Max wheel-torque utilization during dump: {np.max(rho_tau_dump):.4f} "
          f"(tau_max = {wheel.tau_max} N*m) -- "
          f"{'FEASIBLE' if np.max(rho_tau_dump) <= 1.0 else 'INFEASIBLE, reported honestly'}")
    print(f"  For reference, M1's worst maneuver required tau_req = 0.8796 N*m "
          f"(ratio dump/maneuver peak torque = {np.max(np.abs(dump4.tau_w))/0.8796:.2e})")

    # -----------------------------------------------------------------
    # Repeat interval and duty cycle (single-cycle analytical estimate)
    # -----------------------------------------------------------------
    hr()
    print("REPEAT INTERVAL & DUTY CYCLE (single-cycle analytical estimate, nominal 4-wheel)")
    hr()
    worst_idx4 = int(np.argmax(np.abs(tau_w_mean4)))
    t_repeat4 = analytical_repeat_interval(H_on, H_off, tau_w_mean4[worst_idx4])
    duty_cycle4 = dump4.duration / (dump4.duration + t_repeat4)
    print(f"  Limiting wheel: {g4.labels[worst_idx4]}, secular torque = {tau_w_mean4[worst_idx4]:.3e} N*m")
    print(f"  Analytical repeat interval (H_on-H_off band): {t_repeat4:.3e} s "
          f"= {t_repeat4/T_orb:.1f} orbits = {t_repeat4/86400:.2f} days")
    print(f"  Duty cycle = t_dump/(t_dump+T_repeat) = {duty_cycle4*100:.3f}%")
    print(f"  For comparison, M3's zero-to-H_threshold time was 1558.5 orbits; the operational "
          f"H_on-H_off BAND repeat interval ({t_repeat4/T_orb:.1f} orbits) is the band fraction "
          f"({(H_on-H_off)/H_on*100:.0f}% of H_on) of that full-range time, as expected.")

    # -----------------------------------------------------------------
    # Long-duration hybrid mission schedule (nominal 4-wheel, 1 year)
    # -----------------------------------------------------------------
    hr()
    print("LONG-DURATION MISSION SCHEDULE (nominal 4-wheel, representative 1-year horizon)")
    hr()
    T_horizon = 365 * 86400.0
    events4 = simulate_mission_schedule(
        g4, env, tau_w_mean4, B_field, mt, k_H, thresholds, T_horizon,
        dt_dump=dt_dump, t_max_dump=t_max_dump, T_orb=T_orb,
    )
    stats4 = schedule_statistics(events4, T_horizon)
    print(f"  Horizon: {T_horizon/86400:.0f} days ({T_horizon/T_orb:.0f} orbits)")
    print(f"  Total dumps in year 1 (starting from zero wheel momentum): {stats4.n_dumps}")
    print(f"  NOTE: year 1 includes the initial zero-to-H_on charge-up (the full M3-scale accumulation")
    print(f"  time, longer than a steady-state H_off-to-H_on cycle), so this count is LOWER than the")
    print(f"  steady-state average rate (365 days / repeat interval = 365/{t_repeat4/86400:.1f} = "
          f"{365/(t_repeat4/86400):.1f}/year) reported in M5. Both are correct under their own definition")
    print(f"  -- see docs/desaturation_methodology.md for the reconciliation.")
    print(f"  Mean interval: {stats4.mean_interval:.3e} s ({stats4.mean_interval/86400:.2f} days)"
          if stats4.mean_interval else "  Mean interval: n/a (fewer than 2 dumps)")
    print(f"  Min/Max interval: "
          f"{stats4.min_interval/86400:.2f} / {stats4.max_interval/86400:.2f} days"
          if stats4.min_interval else "  Min/Max interval: n/a")
    print(f"  Total desaturation time: {stats4.total_dump_time:.3e} s "
          f"({stats4.total_dump_time/3600:.2f} hours)")
    print(f"  Duty cycle: {stats4.duty_cycle*100:.4f}%")

    # -----------------------------------------------------------------
    # Magnetorquer capability sweep
    # -----------------------------------------------------------------
    hr()
    print("MAGNETORQUER CAPABILITY SWEEP (nominal 4-wheel)")
    hr()
    sweep_factors = [0.5, 1.0, 2.0, 4.0]
    sweep_results = []
    for factor in sweep_factors:
        mt_s = Magnetorquer(m_max=mt.m_max * factor)
        dump_s = simulate_dump(g4, h0_4, env, B_field, mt_s, k_H, H_off, dt_dump, t_max_dump)
        sweep_results.append((factor, dump_s))
        util = np.max(np.linalg.norm(dump_s.m_cmd, axis=1)) / mt_s.m_max
        print(f"  m_max = {factor}x baseline ({mt_s.m_max:.1f} A*m^2): "
              f"reached_off={dump_s.reached_off}, duration={dump_s.duration:.1f} s "
              f"({dump_s.duration/T_orb:.2f} orbits), mean effectiveness={np.mean(dump_s.effectiveness):.3f}, "
              f"peak dipole utilization={util:.3f}")

    # -----------------------------------------------------------------
    # Threshold trade study
    # -----------------------------------------------------------------
    hr()
    print("THRESHOLD (H_on/H_off) TRADE STUDY (nominal 4-wheel)")
    hr()
    threshold_pairs = [(0.9, 0.5), (0.8, 0.4), (0.7, 0.3), (0.8, 0.6)]
    for f_on, f_off in threshold_pairs:
        H_on_s, H_off_s = f_on * H_max, f_off * H_max
        h0_s = natural_h0(g4, tau_w_mean4, H_on_s)
        dump_s = simulate_dump(g4, h0_s, env, B_field, mt, k_H, H_off_s, dt_dump, t_max_dump)
        t_repeat_s = analytical_repeat_interval(H_on_s, H_off_s, tau_w_mean4[worst_idx4])
        duty_s = dump_s.duration / (dump_s.duration + t_repeat_s)
        headroom = (H_max - H_on_s) / H_max
        print(f"  H_on={f_on}*Hmax, H_off={f_off}*Hmax: dump={dump_s.duration/T_orb:.2f} orbits, "
              f"repeat={t_repeat_s/T_orb:.1f} orbits, duty={duty_s*100:.4f}%, "
              f"headroom-to-Hmax={headroom*100:.0f}%")

    # -----------------------------------------------------------------
    # 3-wheel vs 4-wheel comparison
    # -----------------------------------------------------------------
    hr()
    print("3-WHEEL vs 4-WHEEL DESATURATION COMPARISON")
    hr()
    tau_w_mean3 = mean_wheel_torque(g3, env, T_orb, n=4000)
    worst_idx3 = int(np.argmax(np.abs(tau_w_mean3)))
    h0_3 = natural_h0(g3, tau_w_mean3, H_on)
    dump3 = simulate_dump(g3, h0_3, env, B_field, mt, k_H, H_off, dt_dump, t_max_dump)
    t_repeat3 = analytical_repeat_interval(H_on, H_off, tau_w_mean3[worst_idx3])
    duty3 = dump3.duration / (dump3.duration + t_repeat3)
    rho_tau_dump3 = per_wheel_torque_utilization(dump3.tau_w, wheel.tau_max)

    print(f"  3-wheel: dump={dump3.duration/T_orb:.2f} orbits, repeat={t_repeat3/T_orb:.1f} orbits, "
          f"duty={duty3*100:.4f}%, max wheel-torque util={np.max(rho_tau_dump3):.4f}, "
          f"final |h_w| max={np.max(np.abs(dump3.h_w_final)):.4f}")
    print(f"  4-wheel: dump={dump4.duration/T_orb:.2f} orbits, repeat={t_repeat4/T_orb:.1f} orbits, "
          f"duty={duty_cycle4*100:.4f}%, max wheel-torque util={np.max(rho_tau_dump):.4f}, "
          f"final |h_w| max={np.max(np.abs(dump4.h_w_final)):.4f}")
    better = "4-wheel" if (t_repeat4 + 0) > t_repeat3 else "3-wheel"
    print(f"  Longer repeat interval (fewer dumps needed): {better} "
          "(consistent with M3's momentum-accumulation lifetime comparison)")

    # -----------------------------------------------------------------
    # Single-wheel-failure desaturation
    # -----------------------------------------------------------------
    hr()
    print("SINGLE-WHEEL-FAILURE DESATURATION (4-wheel tetrahedral)")
    hr()
    failure_dump_results = {}
    for idx in range(4):
        g_failed = g4.remove_wheel(idx)
        tau_w_mean_f = mean_wheel_torque(g_failed, env, T_orb, n=4000)
        worst_idx_f = int(np.argmax(np.abs(tau_w_mean_f)))
        h0_f = natural_h0(g_failed, tau_w_mean_f, H_on)
        dump_f = simulate_dump(g_failed, h0_f, env, B_field, mt, k_H, H_off, dt_dump, t_max_dump)
        rho_tau_f = per_wheel_torque_utilization(dump_f.tau_w, wheel.tau_max)
        t_repeat_f = analytical_repeat_interval(H_on, H_off, tau_w_mean_f[worst_idx_f])
        failure_dump_results[idx] = dict(dump=dump_f, rho_tau=rho_tau_f, t_repeat=t_repeat_f)
        print(f"  Wheel {g4.labels[idx]} failed: dump={dump_f.duration/T_orb:.2f} orbits, "
              f"reached_off={dump_f.reached_off}, max wheel-torque util={np.max(rho_tau_f):.4f}, "
              f"repeat={t_repeat_f/T_orb:.1f} orbits")

    # -----------------------------------------------------------------
    # Null-space redistribution vs external unloading
    # -----------------------------------------------------------------
    hr()
    print("NULL-SPACE REDISTRIBUTION vs EXTERNAL UNLOADING (4-wheel)")
    hr()
    h_demo = h0_4.copy()
    H_body_before = g4.A @ h_demo
    for z in (0.0, 2.0, -3.0):
        h_redist = null_space_redistribute(g4, h_demo, [z])
        H_body_after = g4.A @ h_redist
        print(f"  z={z:+.1f}: h_w changes to {np.round(h_redist, 3)}, "
              f"A@h_w = {np.round(H_body_after, 6)} "
              f"(unchanged from {np.round(H_body_before, 6)}: "
              f"{'YES' if np.allclose(H_body_after, H_body_before, atol=1e-9) else 'NO -- BUG'})")
    print("  -> Null-space redistribution NEVER changes the body-observable momentum A@h_w: "
          "it cannot, by itself, remove any spacecraft angular momentum.")
    short_dump = simulate_dump(g4, h_demo, env, B_field, mt, k_H, H_off, dt_dump, t_max_dump)
    print(f"  Applying external magnetorquer unloading DOES reduce A@h_w over time: "
          f"|A@h_w| from {np.linalg.norm(g4.A @ h_demo):.4f} to "
          f"{np.linalg.norm(g4.A @ short_dump.h_w_final):.4f} N*m*s over the dump "
          f"-- only an external torque can do this.")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig1 = os.path.join(RESULTS_DIR, "fig1_single_dump_event.png")
    fig2 = os.path.join(RESULTS_DIR, "fig2_magnetic_unloading_geometry.png")
    fig3 = os.path.join(RESULTS_DIR, "fig3_long_duration_cycles.png")
    fig4 = os.path.join(RESULTS_DIR, "fig4_dump_duration_vs_dipole_capability.png")
    fig5 = os.path.join(RESULTS_DIR, "fig5_threshold_band_trade.png")

    make_figure1(dump4, g4, thresholds, fig1)
    make_figure2(dump4, B_field, fig2)
    make_figure3(events4[:3], g4, env, B_field, mt, k_H, thresholds, H_max, T_orb, dt_dump, t_max_dump, fig3)
    make_figure4(sweep_results, T_orb, fig4)
    make_figure5(threshold_pairs, g4, env, B_field, mt, k_H, tau_w_mean4, worst_idx4,
                 H_max, dt_dump, t_max_dump, fig5)

    print(f"\nFigures written to: {RESULTS_DIR}")
    for p in (fig1, fig2, fig3, fig4, fig5):
        print(f"  {os.path.relpath(p)}")

    # -----------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------
    table_path = os.path.join(RESULTS_DIR, "desaturation_table.md")
    write_desaturation_table(mt, H_on, H_off, dump3, dump4, duty3, duty_cycle4,
                              g3, g4, rho_tau_dump3, rho_tau_dump, failure_dump_results,
                              wheel, table_path)
    print(f"\nDesaturation summary table written to: {os.path.relpath(table_path)}")

    csv_path = os.path.join(RESULTS_DIR, "desaturation_schedule.csv")
    write_schedule_csv(events4, T_orb, csv_path)
    print(f"Desaturation schedule CSV written to: {os.path.relpath(csv_path)}")

    hr("=")
    print("END OF M4 ANALYSIS REPORT")
    hr("=")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure1(dump, geometry, thresholds, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for i in range(geometry.n_wheels):
        ax.plot(dump.t / 3600, dump.h_w[:, i], label=f"Wheel {geometry.labels[i]}", color=colors[i])
    ax.axhline(thresholds.H_on, color="red", linestyle="--", linewidth=1.2, label=r"$+H_{on}$")
    ax.axhline(-thresholds.H_on, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(thresholds.H_off, color="green", linestyle="--", linewidth=1.2, label=r"$+H_{off}$")
    ax.axhline(-thresholds.H_off, color="green", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Wheel momentum h_w [N*m*s]")
    ax.set_title("Figure 1 — Single Desaturation Event (nominal 4-wheel)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure2(dump, B_field, path):
    t = dump.t
    B_hist = np.array([B_field(tk) for tk in t])
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax0b = axs[0].twinx()
    l1, = axs[0].plot(t / 3600, np.linalg.norm(B_hist, axis=1) * 1e5, color="tab:blue", label="|B(t)|")
    l2, = ax0b.plot(t / 3600, np.linalg.norm(dump.tau_achieved, axis=1) * 1e4, color="tab:red",
                     label="|tau_achieved|")
    axs[0].set_ylabel(r"|B(t)| [$10^{-5}$ T]", color="tab:blue")
    axs[0].tick_params(axis="y", labelcolor="tab:blue")
    ax0b.set_ylabel(r"|tau_achieved| [$10^{-4}$ N$\cdot$m]", color="tab:red")
    ax0b.tick_params(axis="y", labelcolor="tab:red")
    axs[0].set_title("Field magnitude and achieved unloading torque")
    axs[0].legend(handles=[l1, l2], fontsize=8, loc="upper right")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t / 3600, dump.effectiveness, color="tab:green")
    axs[1].set_ylabel(r"Effectiveness $\eta_B(t)$")
    axs[1].set_xlabel("Time [hours]")
    axs[1].set_title("Field-geometry unloading effectiveness (independent of dipole saturation)")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].grid(True, alpha=0.3)

    fig.suptitle("Figure 2 — Magnetic Unloading Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure3(events, g4, env, B_field, mt, k_H, thresholds, H_max, T_orb, dt_dump, t_max_dump, path):
    """Reconstruct a readable few-cycle utilization sawtooth: an analytic
    linear accumulation ramp (ACCUMULATING, matching the hybrid schedule
    simulation's own approximation) followed by the ACTUAL closed-loop
    dump trace (re-run from each event's recorded initial condition,
    DESATURATING) for the first few mission-schedule events. Plotting the
    full 1-year/6-dump horizon on one axis makes each ~3-orbit dump
    invisible against a ~780-orbit repeat interval -- a few zoomed cycles
    tell the sawtooth story far more clearly.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    orbit_offset = 0.0
    for i, ev in enumerate(events):
        # Accumulation ramp: from this event's own recorded pre-dump utilization
        # trajectory is not stored pointwise, but since accumulation is LINEAR
        # (constant secular torque, per the hybrid approximation), a straight
        # line from the previous event's final utilization up to this event's
        # initial utilization is the exact reconstruction.
        if i == 0:
            ramp_start_util = ev.final_utilization  # approximate start at a released state
            ramp_start_orbit = ev.start_orbit - (ev.start_orbit * 0.3)
        else:
            ramp_start_util = events[i - 1].final_utilization
            ramp_start_orbit = events[i - 1].start_orbit + events[i - 1].duration / T_orb
        ramp_end_orbit = ev.start_orbit
        ax.plot([ramp_start_orbit, ramp_end_orbit],
                [ramp_start_util * thresholds.H_on / H_max, ev.initial_utilization * thresholds.H_on / H_max],
                color="tab:blue", linewidth=1.5)

        # Real dump trace (re-simulated from the recorded initial condition).
        def tau_d_shifted(tau_local, t0=ev.start_time):
            return env(t0 + tau_local)

        def B_field_shifted(tau_local, t0=ev.start_time):
            return B_field(t0 + tau_local)

        dump = simulate_dump(g4, ev.h_w_initial, tau_d_shifted, B_field_shifted, mt, k_H,
                              thresholds.H_off, dt_dump, min(t_max_dump, ev.duration * 1.05 + dt_dump))
        util_trace = np.max(np.abs(dump.h_w), axis=1) / H_max
        ax.plot(ev.start_orbit + dump.t / T_orb, util_trace, color="tab:red", linewidth=1.8)
        ax.axvspan(ev.start_orbit, ev.start_orbit + ev.duration / T_orb, color="tab:red", alpha=0.15)

    ax.axhline(thresholds.H_on / H_max, color="black", linestyle="--", linewidth=1.0, label=r"$H_{on}/H_{max}$")
    ax.axhline(thresholds.H_off / H_max, color="gray", linestyle="--", linewidth=1.0, label=r"$H_{off}/H_{max}$")
    ax.plot([], [], color="tab:blue", label="Accumulating (analytic ramp)")
    ax.plot([], [], color="tab:red", label="Desaturating (closed-loop)")
    ax.set_xlabel("Orbits")
    ax.set_ylabel(r"Worst-wheel utilization $\max_i|h_i|/H_{max}$")
    ax.set_title("Figure 3 — Long-Duration Momentum-Management Cycles (first 3 events)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure4(sweep_results, T_orb, path):
    factors = [f for f, _ in sweep_results]
    durations = [d.duration / T_orb for _, d in sweep_results]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(factors, durations, "o-", color="tab:blue")
    ax.set_xlabel("Magnetorquer dipole capability (x baseline m_max)")
    ax.set_ylabel("Dump duration [orbits]")
    ax.set_title("Figure 4 — Dump Duration vs Magnetorquer Dipole Capability")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure5(threshold_pairs, g4, env, B_field, mt, k_H, tau_w_mean4, worst_idx4,
                  H_max, dt_dump, t_max_dump, path):
    labels = []
    durations = []
    duties = []
    for f_on, f_off in threshold_pairs:
        H_on_s, H_off_s = f_on * H_max, f_off * H_max
        h0_s = tau_w_mean4 / np.max(np.abs(tau_w_mean4)) * H_on_s
        dump_s = simulate_dump(g4, h0_s, env, B_field, mt, k_H, H_off_s, dt_dump, t_max_dump)
        t_repeat_s = analytical_repeat_interval(H_on_s, H_off_s, tau_w_mean4[worst_idx4])
        duty_s = dump_s.duration / (dump_s.duration + t_repeat_s)
        labels.append(f"{f_on}/{f_off}")
        durations.append(t_repeat_s / (5677.0))  # orbits, approx T_orb fallback
        duties.append(duty_s * 100)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    axs[0].bar(labels, durations, color="tab:blue")
    axs[0].set_ylabel("Repeat interval [orbits]")
    axs[0].set_xlabel(r"$H_{on}/H_{max}$ / $H_{off}/H_{max}$")
    axs[0].set_title("Repeat interval vs threshold band")
    axs[0].grid(True, alpha=0.3, axis="y")

    axs[1].bar(labels, duties, color="tab:orange")
    axs[1].set_ylabel("Duty cycle [%]")
    axs[1].set_xlabel(r"$H_{on}/H_{max}$ / $H_{off}/H_{max}$")
    axs[1].set_title("Duty cycle vs threshold band")
    axs[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Figure 5 — Threshold-Band Trade")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_desaturation_table(mt, H_on, H_off, dump3, dump4, duty3, duty4, g3, g4,
                              rho_tau3, rho_tau4, failure_results, wheel, path):
    lines = ["# Milestone 4 — Desaturation Summary Table\n"]
    lines.append(
        f"Magnetorquer: m_max = {mt.m_max} A*m^2. H_on = {H_on:.4f} N*m*s, "
        f"H_off = {H_off:.4f} N*m*s.\n"
    )
    lines.append(
        "| Configuration | Dump duration [orbits] | Max wheel-torque util | "
        "Max dipole util | Worst wheel |"
    )
    lines.append("|---|---|---|---|---|")
    T_orb_local = 5677.0
    lines.append(
        f"| 3-wheel orthogonal (nominal) | {dump3.duration/T_orb_local:.2f} | "
        f"{np.max(rho_tau3):.4f} | {np.max(np.linalg.norm(dump3.m_cmd,axis=1))/mt.m_max:.3f} | "
        f"{g3.labels[int(np.argmax(np.max(np.abs(dump3.h_w),axis=0)))]} |"
    )
    lines.append(
        f"| 4-wheel tetrahedral (nominal) | {dump4.duration/T_orb_local:.2f} | "
        f"{np.max(rho_tau4):.4f} | {np.max(np.linalg.norm(dump4.m_cmd,axis=1))/mt.m_max:.3f} | "
        f"{g4.labels[int(np.argmax(np.max(np.abs(dump4.h_w),axis=0)))]} |"
    )
    for idx in range(4):
        d = failure_results[idx]["dump"]
        rt = failure_results[idx]["rho_tau"]
        lines.append(
            f"| 4-wheel, {g4.labels[idx]} failed | {d.duration/T_orb_local:.2f} | "
            f"{np.max(rt):.4f} | {np.max(np.linalg.norm(d.m_cmd,axis=1))/mt.m_max:.3f} | "
            f"(surviving wheels) |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_schedule_csv(events, T_orb, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dump_number", "start_time_s", "start_orbit", "limiting_wheel",
            "initial_utilization", "duration_s", "duration_orbits",
            "final_utilization", "next_interval_s", "next_interval_orbits",
        ])
        for ev in events:
            writer.writerow([
                ev.dump_number, f"{ev.start_time:.1f}", f"{ev.start_orbit:.2f}",
                ev.limiting_wheel, f"{ev.initial_utilization:.4f}",
                f"{ev.duration:.1f}", f"{ev.duration/T_orb:.3f}",
                f"{ev.final_utilization:.4f}",
                f"{ev.next_interval:.1f}" if ev.next_interval is not None else "",
                f"{ev.next_interval/T_orb:.2f}" if ev.next_interval is not None else "",
            ])


if __name__ == "__main__":
    main()

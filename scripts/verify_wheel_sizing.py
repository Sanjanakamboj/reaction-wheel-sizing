#!/usr/bin/env python3
"""
Milestone 1 verification script.

Prints a concise engineering report (spacecraft inertia, representative
wheel, maneuver sizing, analytical-vs-numerical residuals, momentum
conservation, scaling-law checks) and generates:

    results/fig1_rest_to_rest_maneuver.png
    results/fig2_momentum_exchange.png
    results/fig3_torque_momentum_vs_duration.png
    results/maneuver_sizing_table.md

Run with:  python scripts/verify_wheel_sizing.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.wheel import representative_wheel
from reaction_wheel.maneuvers import (
    triangular_slew_requirement,
    propagate_rest_to_rest_slew,
)
from reaction_wheel.sizing import size_single_axis_maneuver, SizingMargins

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Representative maneuver cases: (label, angle_deg, duration_s)
MANEUVER_CASES = [
    ("10 deg / 20 s", 10.0, 20.0),
    ("30 deg / 30 s", 30.0, 30.0),
    ("60 deg / 60 s", 60.0, 60.0),
    ("90 deg / 60 s", 90.0, 60.0),
    ("45 deg / 10 s (aggressive)", 45.0, 10.0),
]


def hr(char="-", n=78):
    print(char * n)


def main():
    sc = representative_spacecraft()
    wheel = representative_wheel()
    margins = SizingMargins()

    hr("=")
    print("MILESTONE 1 VERIFICATION REPORT — Reaction-Wheel Mechanics & Maneuver Sizing")
    hr("=")

    print(f"\nRepresentative spacecraft: {sc.name}")
    print(f"  Ix = {sc.Ix:.3f} kg*m^2")
    print(f"  Iy = {sc.Iy:.3f} kg*m^2")
    print(f"  Iz = {sc.Iz:.3f} kg*m^2")
    print(f"  Positive definite: {sc.is_positive_definite()}")

    print(f"\nRepresentative reaction wheel: {wheel.name}")
    print(f"  J_w       = {wheel.J_w:.4f} kg*m^2")
    print(f"  Omega_max = {wheel.Omega_max:.2f} rad/s ({wheel.Omega_max_rpm:.0f} rpm)")
    print(f"  tau_max   = {wheel.tau_max:.3f} N*m")
    print(f"  H_max     = {wheel.H_max:.4f} N*m*s")
    print(f"\nMargins: SF_tau = {margins.SF_tau}, SF_H = {margins.SF_H} (illustrative, see docs)")

    # -----------------------------------------------------------------
    # Maneuver-sizing table across axes and cases
    # -----------------------------------------------------------------
    axes = ["x", "y", "z"]
    rows = []
    for label, theta_deg, T in MANEUVER_CASES:
        for axis in axes:
            I = sc.inertia_about(axis)
            result = size_single_axis_maneuver(
                axis, theta_deg, T, I, J_w_ref=wheel.J_w,
                margins=margins, candidate=wheel,
            )
            rows.append((label, theta_deg, T, axis, I, result))

    hr()
    print("MANEUVER-SIZING SUMMARY (candidate = representative wheel)")
    hr()
    header = (
        f"{'Case':<26}{'Axis':<5}{'I[kg m2]':>10}{'alpha[rad/s2]':>15}"
        f"{'omega_pk[rad/s]':>17}{'tau_req[Nm]':>13}{'H_req[Nms]':>12}"
        f"{'Omega_req[rpm]':>15}{'rho_tau':>9}{'rho_H':>9}{'driver':>16}"
    )
    print(header)
    for label, theta_deg, T, axis, I, r in rows:
        print(
            f"{label:<26}{axis:<5}{I:>10.3f}{r.alpha:>15.5f}"
            f"{r.omega_peak:>17.5f}{r.tau_req:>13.5f}{r.H_req:>12.5f}"
            f"{r.Omega_req_rpm:>15.2f}{r.rho_tau:>9.3f}{r.rho_H:>9.3f}"
            f"{r.active_constraint.value:>16}"
        )

    # Identify overall peak torque / peak momentum drivers
    peak_tau_row = max(rows, key=lambda rr: rr[5].tau_req)
    peak_H_row = max(rows, key=lambda rr: rr[5].H_req)
    worst_axis_by_inertia = max(axes, key=sc.inertia_about)

    hr()
    print("KEY ENGINEERING FINDINGS")
    hr()
    print(f"  Peak-torque driver:    {peak_tau_row[0]} about axis {peak_tau_row[3]} "
          f"-> tau_req = {peak_tau_row[5].tau_req:.4f} N*m")
    print(f"  Peak-momentum driver:  {peak_H_row[0]} about axis {peak_H_row[3]} "
          f"-> H_req = {peak_H_row[5].H_req:.4f} N*m*s")
    print(f"  Worst axis by inertia: {worst_axis_by_inertia} "
          f"(I = {sc.inertia_about(worst_axis_by_inertia):.3f} kg*m^2)")

    # -----------------------------------------------------------------
    # Analytical vs numerical verification, and momentum conservation,
    # for one representative case (60 deg / 60 s about the y axis).
    # -----------------------------------------------------------------
    hr()
    print("ANALYTICAL vs NUMERICAL VERIFICATION (60 deg / 60 s, y-axis)")
    hr()
    I_y = sc.Iy
    theta_rad = np.deg2rad(60.0)
    T = 60.0
    req = triangular_slew_requirement(theta_rad, T, I_y)
    hist = propagate_rest_to_rest_slew(theta_rad, T, I_y, n_points=4000)

    final_theta_residual = abs(hist.theta[-1] - theta_rad)
    final_omega_residual = abs(hist.omega[-1])
    mid_idx = np.argmin(np.abs(hist.t - T / 2.0))
    peak_omega_residual = abs(abs(hist.omega[mid_idx]) - req.omega_peak)
    peak_H_residual = abs(np.max(np.abs(hist.H_wheel)) - req.H_body_peak)

    H_total = I_y * hist.omega + hist.H_wheel
    conservation_residual = np.max(np.abs(H_total))

    # Independent path: integrate wheel torque over the accel phase.
    tau_w_cmd = req.wheel_torque_cmd
    dt = hist.t[1] - hist.t[0]
    accel_mask = hist.t <= T / 2.0
    delta_H_w_integrated = np.trapezoid(
        np.full(np.count_nonzero(accel_mask), tau_w_cmd), dx=dt
    )
    momentum_integration_residual = abs(abs(delta_H_w_integrated) - req.H_body_peak)

    print(f"  Final attitude angle residual:   {final_theta_residual:.3e} rad")
    print(f"  Final body-rate residual:        {final_omega_residual:.3e} rad/s")
    print(f"  Peak body-rate residual:         {peak_omega_residual:.3e} rad/s")
    print(f"  Peak wheel-momentum residual:    {peak_H_residual:.3e} N*m*s")
    print(f"  Total-momentum conservation max|H_total|: {conservation_residual:.3e} N*m*s")
    print(f"  Torque-integration vs momentum-requirement residual: "
          f"{momentum_integration_residual:.3e} N*m*s")

    # -----------------------------------------------------------------
    # Scaling-law checks
    # -----------------------------------------------------------------
    hr()
    print("SCALING-LAW CHECKS (fixed I = Iy)")
    hr()
    base = triangular_slew_requirement(np.deg2rad(30.0), 30.0, I_y)
    doubled_angle = triangular_slew_requirement(np.deg2rad(60.0), 30.0, I_y)
    doubled_time = triangular_slew_requirement(np.deg2rad(30.0), 60.0, I_y)
    print(f"  Doubling angle at fixed T:  tau ratio = {doubled_angle.tau_req/base.tau_req:.4f} "
          f"(expect 2.0000);  H ratio = {doubled_angle.H_body_peak/base.H_body_peak:.4f} (expect 2.0000)")
    print(f"  Doubling time at fixed theta: tau ratio = {doubled_time.tau_req/base.tau_req:.4f} "
          f"(expect 0.2500);  H ratio = {doubled_time.H_body_peak/base.H_body_peak:.4f} (expect 0.5000)")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig1_path = os.path.join(RESULTS_DIR, "fig1_rest_to_rest_maneuver.png")
    fig2_path = os.path.join(RESULTS_DIR, "fig2_momentum_exchange.png")
    fig3_path = os.path.join(RESULTS_DIR, "fig3_torque_momentum_vs_duration.png")

    make_figure1(hist, fig1_path)
    make_figure2(hist, I_y, fig2_path)
    make_figure3(I_y, fig3_path)

    print(f"\nFigures written to: {RESULTS_DIR}")
    for p in (fig1_path, fig2_path, fig3_path):
        print(f"  {os.path.relpath(p)}")

    # -----------------------------------------------------------------
    # Markdown maneuver-sizing table
    # -----------------------------------------------------------------
    table_path = os.path.join(RESULTS_DIR, "maneuver_sizing_table.md")
    write_table(rows, wheel, margins, table_path)
    print(f"\nManeuver-sizing table written to: {os.path.relpath(table_path)}")

    hr("=")
    print("END OF M1 VERIFICATION REPORT")
    hr("=")


def make_figure1(hist, path):
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    axs[0].plot(hist.t, np.rad2deg(hist.theta), color="tab:blue")
    axs[0].set_ylabel("Attitude angle [deg]")
    axs[0].set_title("Figure 1 — Rest-to-Rest Maneuver (60 deg / 60 s, y-axis)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(hist.t, np.rad2deg(hist.omega), color="tab:orange")
    axs[1].set_ylabel("Body rate [deg/s]")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(hist.t, hist.tau_body, color="tab:green")
    axs[2].set_ylabel("Applied body torque [N*m]")
    axs[2].set_xlabel("Time [s]")
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure2(hist, I, path):
    H_body = I * hist.omega
    H_total = H_body + hist.H_wheel

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist.t, H_body, label="Spacecraft body $H_{body}$", color="tab:blue")
    ax.plot(hist.t, hist.H_wheel, label="Wheel $H_w$", color="tab:orange")
    ax.plot(hist.t, H_total, label="Total $H_{total}$", color="tab:red", linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angular momentum [N*m*s]")
    ax.set_title("Figure 2 — Momentum Exchange (isolated system)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure3(I, path):
    theta_fixed_deg = 30.0
    T_range = np.linspace(10.0, 120.0, 200)
    tau_vals = np.array([
        triangular_slew_requirement(np.deg2rad(theta_fixed_deg), T, I).tau_req
        for T in T_range
    ])
    H_vals = np.array([
        triangular_slew_requirement(np.deg2rad(theta_fixed_deg), T, I).H_body_peak
        for T in T_range
    ])

    # Normalize both curves to their value at T_range[0] so the DIFFERENT
    # power-law slopes (T^-2 vs T^-1) are directly visually comparable on
    # one shared log-log axis (a dual-axis plot would auto-scale each
    # curve independently and hide the slope difference).
    tau_norm = tau_vals / tau_vals[0]
    H_norm = H_vals / H_vals[0]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    ax1 = axs[0]
    ax1.plot(T_range, tau_norm, color="tab:blue", label=r"$\tau_{req}/\tau_{req}(T_0) \propto T^{-2}$")
    ax1.plot(T_range, H_norm, color="tab:orange", label=r"$H_{req}/H_{req}(T_0) \propto T^{-1}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Maneuver duration T [s]")
    ax1.set_ylabel(r"Requirement normalized to value at $T_0$ = 10 s")
    ax1.set_title("Normalized comparison (equal footing)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()

    ax2 = axs[1]
    ax2.plot(T_range, tau_vals, color="tab:blue", label=r"$\tau_{req}$ [N·m]")
    ax2.set_xlabel("Maneuver duration T [s]")
    ax2.set_ylabel("Required torque [N*m]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3, which="both")
    ax2b = ax2.twinx()
    ax2b.plot(T_range, H_vals, color="tab:orange", label=r"$H_{req}$ [N·m·s]")
    ax2b.set_ylabel("Required momentum excursion [N*m*s]", color="tab:orange")
    ax2b.tick_params(axis="y", labelcolor="tab:orange")
    ax2b.set_yscale("log")
    ax2.set_title("Absolute values (engineering units)")

    fig.suptitle(f"Figure 3 — Torque/Momentum Requirement vs Maneuver Duration ({theta_fixed_deg} deg slew)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(rows, wheel, margins, path):
    lines = []
    lines.append("# Milestone 1 — Maneuver Sizing Table\n")
    lines.append(
        f"Candidate wheel: **{wheel.name}** "
        f"(J_w = {wheel.J_w} kg*m^2, tau_max = {wheel.tau_max} N*m, "
        f"Omega_max = {wheel.Omega_max_rpm:.0f} rpm, H_max = {wheel.H_max:.4f} N*m*s).\n"
    )
    lines.append(
        f"Margins: SF_tau = {margins.SF_tau}, SF_H = {margins.SF_H} "
        "(illustrative, not adopted mission requirements — see docs/wheel_sizing_methodology.md).\n"
    )
    lines.append(
        "| Case | Axis | I [kg·m²] | α [rad/s²] | ω_peak [rad/s] | τ_req [N·m] | "
        "H_req [N·m·s] | Ω_req [rpm] | τ_sized [N·m] | H_sized [N·m·s] | ρ_τ | ρ_H | Active constraint |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, theta_deg, T, axis, I, r in rows:
        lines.append(
            f"| {label} | {axis} | {I:.3f} | {r.alpha:.5f} | {r.omega_peak:.5f} | "
            f"{r.tau_req:.5f} | {r.H_req:.5f} | {r.Omega_req_rpm:.2f} | "
            f"{r.tau_sized:.5f} | {r.H_sized:.5f} | {r.rho_tau:.3f} | {r.rho_H:.3f} | "
            f"{r.active_constraint.value} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

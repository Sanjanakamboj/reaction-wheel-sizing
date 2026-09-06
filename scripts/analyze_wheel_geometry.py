#!/usr/bin/env python3
"""
Milestone 2 analysis script.

Prints a concise geometry/allocation/redundancy report and generates:

    results/fig1_wheel_axis_geometry.png
    results/fig2_wheel_torque_allocations.png
    results/fig3_directional_torque_capability.png
    results/fig4_nominal_vs_failed_capability.png
    results/wheel_loading_table.md

Run with:  python scripts/analyze_wheel_geometry.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.wheel import representative_wheel
from reaction_wheel.maneuvers import triangular_slew_requirement
from reaction_wheel.geometry import (
    orthogonal_3wheel,
    tetrahedral_4wheel,
    allocate_torque,
    allocate_momentum,
    null_space_alternative_allocation,
    per_wheel_torque_utilization,
    per_wheel_momentum_utilization,
    is_torque_allocation_feasible,
    describe_feasibility,
    directional_capability_envelope,
    isotropy_ratio,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hr(char="-", n=86):
    print(char * n)


def main():
    sc = representative_spacecraft()
    wheel = representative_wheel()
    tau_max = wheel.tau_max
    H_max = wheel.H_max

    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()

    hr("=")
    print("MILESTONE 2 ANALYSIS REPORT — 3-Axis Wheel-Set Geometry, Allocation & Redundancy")
    hr("=")

    print("\nSpacecraft inertia (from M1):")
    print(f"  Ix = {sc.Ix:.3f} kg*m^2, Iy = {sc.Iy:.3f} kg*m^2, Iz = {sc.Iz:.3f} kg*m^2")
    print(f"\nRepresentative wheel capability (identical for every wheel in the set):")
    print(f"  tau_max = {tau_max:.3f} N*m, H_max = {H_max:.4f} N*m*s, J_w = {wheel.J_w:.4f} kg*m^2")

    # -----------------------------------------------------------------
    # Geometry matrices, rank, singular values, condition numbers
    # -----------------------------------------------------------------
    hr()
    print("WHEEL-SET GEOMETRIES")
    hr()
    print("\n3-wheel orthogonal geometry A_3 (columns = wheel spin axes, body frame):")
    print(np.array2string(g3.A, precision=4, suppress_small=True))
    print(f"  rank = {g3.rank()}, singular values = {np.round(g3.singular_values(), 4)}, "
          f"condition number = {g3.condition_number():.4f}")

    print("\n4-wheel tetrahedral geometry A_4:")
    print(np.array2string(g4.A, precision=4, suppress_small=True))
    print(f"  rank = {g4.rank()}, singular values = {np.round(g4.singular_values(), 4)}, "
          f"condition number = {g4.condition_number():.4f}")
    print(f"  null-space dimension = {g4.null_space_dimension()}")
    print(f"  null-space basis (per wheel, up to scale):\n"
          f"  {np.round(g4.null_space_basis().flatten(), 4)}")

    # -----------------------------------------------------------------
    # M1 regression check
    # -----------------------------------------------------------------
    hr()
    print("M1 REGRESSION CHECK (orthogonal 3-wheel geometry)")
    hr()
    theta_agg_deg, T_agg = 45.0, 10.0
    req_y = triangular_slew_requirement(np.deg2rad(theta_agg_deg), T_agg, sc.Iy)
    tau_c_regress = np.array([0.0, req_y.tau_req, 0.0])
    res_regress = allocate_torque(g3, tau_c_regress)
    regress_ok = np.isclose(abs(res_regress.wheel_values[1]), req_y.tau_req, rtol=1e-9)
    print(f"  M1 aggressive case (45deg/10s, y-axis): tau_req = {req_y.tau_req:.4f} N*m")
    print(f"  Orthogonal-geometry wheel torques: {np.round(res_regress.wheel_values, 4)} N*m")
    print(f"  Only wheel Wy is active (regression against M1 single-axis result): "
          f"{'PASS' if regress_ok else 'CHECK'}")
    print(f"  Reconstruction residual: {res_regress.residual_norm:.3e} N*m")

    # -----------------------------------------------------------------
    # Representative torque/momentum demand cases
    # -----------------------------------------------------------------
    hr()
    print("REPRESENTATIVE BODY-TORQUE / MOMENTUM DEMAND CASES")
    hr()

    tau_peak = req_y.tau_req  # 0.8796 N*m, M1's worst-case magnitude
    T_ref = T_agg             # use same maneuver duration to relate H to tau (H = tau * T/2)

    cases = {}
    cases["A/D: worst M1 maneuver (pure y-axis)"] = np.array([0.0, tau_peak, 0.0])
    u_equal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    cases["B: equal-axis demand"] = tau_peak * u_equal
    u_mixed = np.array([1.0, -0.5, 0.8])
    u_mixed = u_mixed / np.linalg.norm(u_mixed)
    cases["C: mixed-axis demand"] = tau_peak * u_mixed

    # Case E: combined 3-axis slew, decoupled principal-axis SIZING APPROXIMATION
    # (NOT exact nonlinear rigid-body attitude dynamics -- see docs).
    theta_vec_deg = np.array([20.0, 30.0, 15.0])
    T_combined = 30.0
    tau_combined = 4.0 * np.array([sc.Ix, sc.Iy, sc.Iz]) * np.deg2rad(theta_vec_deg) / T_combined**2
    cases["E: combined 3-axis slew (sizing approx.)"] = tau_combined

    case_T = {  # maneuver duration associated with each case, for H = tau*(T/2)
        "A/D: worst M1 maneuver (pure y-axis)": T_ref,
        "B: equal-axis demand": T_ref,
        "C: mixed-axis demand": T_ref,
        "E: combined 3-axis slew (sizing approx.)": T_combined,
    }

    table_rows = []
    for label, tau_c in cases.items():
        T_case = case_T[label]
        dH_c = tau_c * (T_case / 2.0)  # H_peak = tau_req * T/2 (triangular-rate identity, per axis)
        for gname, g in (("3-wheel orthogonal", g3), ("4-wheel tetrahedral", g4)):
            res_tau = allocate_torque(g, tau_c)
            res_H = allocate_momentum(g, dH_c)
            rho_tau = per_wheel_torque_utilization(res_tau.wheel_values, tau_max)
            rho_H = per_wheel_momentum_utilization(res_H.wheel_values, H_max)
            worst_tau_idx = int(np.argmax(rho_tau))
            worst_H_idx = int(np.argmax(rho_H))
            feasible = is_torque_allocation_feasible(res_tau.wheel_values, tau_max)
            table_rows.append(dict(
                case=label, geometry=gname, tau_c=tau_c, dH_c=dH_c,
                tau_w=res_tau.wheel_values, H_w=res_H.wheel_values,
                rho_tau_max=float(np.max(rho_tau)), rho_H_max=float(np.max(rho_H)),
                worst_tau_wheel=g.labels[worst_tau_idx], worst_H_wheel=g.labels[worst_H_idx],
                feasible=feasible,
            ))
            print(f"\n  [{label}] geometry={gname}")
            print(f"    tau_c = {np.round(tau_c, 4)} N*m   |tau_c| = {np.linalg.norm(tau_c):.4f} N*m")
            print(f"    tau_w = {np.round(res_tau.wheel_values, 4)} N*m  "
                  f"(worst wheel {g.labels[worst_tau_idx]}, rho_tau_max = {np.max(rho_tau):.3f})")
            print(f"    H_w   = {np.round(res_H.wheel_values, 4)} N*m*s "
                  f"(worst wheel {g.labels[worst_H_idx]}, rho_H_max = {np.max(rho_H):.3f})")
            print(f"    {describe_feasibility(res_tau.wheel_values, tau_max)}")

    # -----------------------------------------------------------------
    # Null-space demonstration (4-wheel)
    # -----------------------------------------------------------------
    hr()
    print("NULL-SPACE DEMONSTRATION (4-wheel tetrahedral)")
    hr()
    tau_c_demo = cases["C: mixed-axis demand"]
    base_alloc = allocate_torque(g4, tau_c_demo).wheel_values
    print(f"  Base minimum-norm allocation: {np.round(base_alloc, 4)} N*m")
    for z in (0.0, 0.05, -0.1):
        alt = null_space_alternative_allocation(g4, tau_c_demo, [z])
        recon = -g4.A @ alt
        print(f"  z = {z:+.2f}: tau_w = {np.round(alt, 4)} N*m  "
              f"-> reconstructed body torque = {np.round(recon, 4)} N*m "
              f"(matches demand to {np.max(np.abs(recon - tau_c_demo)):.2e})")
    print("  The 1-D null space is a spare allocation freedom not used for anything in M2 "
          "(no wheel-speed balancing, no momentum redistribution, no desaturation here) -- "
          "it is only demonstrated to exist for later milestones.")

    # -----------------------------------------------------------------
    # Directional capability envelope + isotropy
    # -----------------------------------------------------------------
    hr()
    print("DIRECTIONAL TORQUE-CAPABILITY ENVELOPE & ISOTROPY")
    hr()
    n_dir = 4000
    iso3 = isotropy_ratio(g3, tau_max, n_directions=n_dir)
    iso4 = isotropy_ratio(g4, tau_max, n_directions=n_dir)
    print(f"  3-wheel orthogonal: tau_min = {iso3['tau_min']:.4f} N*m, "
          f"tau_max_capability = {iso3['tau_max_capability']:.4f} N*m, eta = {iso3['eta']:.4f}")
    print(f"  4-wheel tetrahedral: tau_min = {iso4['tau_min']:.4f} N*m, "
          f"tau_max_capability = {iso4['tau_max_capability']:.4f} N*m, eta = {iso4['eta']:.4f}")
    more_isotropic = "4-wheel tetrahedral" if iso4["eta"] > iso3["eta"] else "3-wheel orthogonal"
    print(f"  More isotropic configuration: {more_isotropic} (higher eta = tau_min/tau_max_capability)")

    # -----------------------------------------------------------------
    # Single-wheel-failure analysis (4-wheel)
    # -----------------------------------------------------------------
    hr()
    print("SINGLE-WHEEL-FAILURE ANALYSIS (4-wheel tetrahedral)")
    hr()
    failure_results = {}
    for idx in range(4):
        g_failed = g4.remove_wheel(idx)
        rank_ok = g_failed.is_full_row_rank()
        iso_failed = isotropy_ratio(g_failed, tau_max, n_directions=n_dir)
        kappa = g_failed.condition_number()
        failure_results[idx] = dict(geometry=g_failed, rank_ok=rank_ok, iso=iso_failed, kappa=kappa)
        print(f"  Wheel {g4.labels[idx]} failed -> remaining rank = {g_failed.rank()} "
              f"(full 3-axis authority: {rank_ok}), "
              f"tau_min = {iso_failed['tau_min']:.4f} N*m, "
              f"tau_max_cap = {iso_failed['tau_max_capability']:.4f} N*m, "
              f"eta = {iso_failed['eta']:.4f}, kappa(A_failed) = {kappa:.4f}")

    failed_tau_mins = np.array([failure_results[i]["iso"]["tau_min"] for i in range(4)])
    print(f"\n  Spread across the 4 single-failure tau_min values: "
          f"{np.max(failed_tau_mins) - np.min(failed_tau_mins):.2e} N*m "
          "(near-zero confirms tetrahedral symmetry: all 4 failure cases are geometrically equivalent)")

    hr()
    print("NOMINAL vs FAILED CAPABILITY (redundancy penalty)")
    hr()
    mean_failed_tau_min = float(np.mean(failed_tau_mins))
    degradation_ratio = mean_failed_tau_min / iso4["tau_min"]
    pct_loss = (1.0 - degradation_ratio) * 100.0
    print(f"  Nominal 4-wheel minimum directional torque capability: {iso4['tau_min']:.4f} N*m")
    print(f"  Mean single-failure (3-wheel-remaining) minimum capability: {mean_failed_tau_min:.4f} N*m")
    print(f"  Degradation ratio (failed/nominal): {degradation_ratio:.4f}")
    print(f"  Percentage capability loss after one wheel fails: {pct_loss:.1f}%")
    print(f"  For reference, 3-wheel orthogonal nominal tau_min: {iso3['tau_min']:.4f} N*m")
    print("  NOTE: redundancy (4th wheel) and increased nominal capability are NOT the same "
          "concept -- see docs/wheel_geometry_methodology.md section on this distinction.")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig1 = os.path.join(RESULTS_DIR, "fig1_wheel_axis_geometry.png")
    fig2 = os.path.join(RESULTS_DIR, "fig2_wheel_torque_allocations.png")
    fig3 = os.path.join(RESULTS_DIR, "fig3_directional_torque_capability.png")
    fig4 = os.path.join(RESULTS_DIR, "fig4_nominal_vs_failed_capability.png")

    make_figure1_geometry(g3, g4, fig1)
    make_figure2_allocations(table_rows, g3, g4, fig2)
    make_figure3_directional_capability(g3, g4, tau_max, fig3)
    make_figure4_failure_capability(g4, failure_results, iso4, tau_max, fig4)

    print(f"\nFigures written to: {RESULTS_DIR}")
    for p in (fig1, fig2, fig3, fig4):
        print(f"  {os.path.relpath(p)}")

    table_path = os.path.join(RESULTS_DIR, "wheel_loading_table.md")
    write_table(table_rows, wheel, table_path)
    print(f"\nWheel-loading table written to: {os.path.relpath(table_path)}")

    hr("=")
    print("END OF M2 ANALYSIS REPORT")
    hr("=")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure1_geometry(g3, g4, path):
    fig = plt.figure(figsize=(11, 5.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_geometry(ax1, g3, "3-Wheel Orthogonal", colors=["tab:red", "tab:green", "tab:blue"])

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_geometry(ax2, g4, "4-Wheel Tetrahedral", colors=["tab:orange", "tab:purple", "tab:brown", "tab:cyan"])

    fig.suptitle("Figure 1 — Wheel-Axis Geometry (body frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_geometry(ax, geometry, title, colors):
    # Body axes: short, light-gray dashed reference arrows so they never
    # visually collide with a wheel axis label even when a wheel is
    # exactly body-axis-aligned (e.g. the orthogonal geometry's Wz || z).
    for vec, lbl in zip(np.eye(3), ("x", "y", "z")):
        ax.quiver(0, 0, 0, *(vec * 0.75), color="gray", linewidth=1.0,
                  arrow_length_ratio=0.15, linestyle="dashed")
        ax.text(*(vec * 0.6), lbl, color="gray", fontsize=9)

    for i in range(geometry.n_wheels):
        a = geometry.A[:, i]
        ax.quiver(0, 0, 0, *a, color=colors[i % len(colors)], linewidth=2.5, arrow_length_ratio=0.15)
        # Push the label further out along the vector so it clears the
        # body-axis labels regardless of viewing angle / foreshortening.
        ax.text(*(a * 1.25), geometry.labels[i], color=colors[i % len(colors)],
                fontsize=11, fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_xlabel("Body x")
    ax.set_ylabel("Body y")
    ax.set_zlabel("Body z")
    ax.set_title(title)
    ax.view_init(elev=18, azim=35)


def make_figure2_allocations(table_rows, g3, g4, path):
    # Grouped bar chart: wheel torque per wheel, for each case, geometry side-by-side.
    cases_order = ["A/D: worst M1 maneuver (pure y-axis)", "B: equal-axis demand",
                   "C: mixed-axis demand", "E: combined 3-axis slew (sizing approx.)"]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (gname, g) in zip(axs, (("3-wheel orthogonal", g3), ("4-wheel tetrahedral", g4))):
        n_wheels = g.n_wheels
        width = 0.8 / len(cases_order)
        x = np.arange(n_wheels)
        for k, case in enumerate(cases_order):
            row = next(r for r in table_rows if r["case"] == case and r["geometry"] == gname)
            offset = (k - (len(cases_order) - 1) / 2.0) * width
            ax.bar(x + offset, row["tau_w"], width=width, label=case.split(":")[0])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(g.labels)
        ax.set_xlabel("Wheel")
        ax.set_ylabel("Wheel torque [N*m]")
        ax.set_title(gname)
        ax.grid(True, alpha=0.3, axis="y")

    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Figure 2 — Representative Wheel Torque Allocations by Case")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure3_directional_capability(g3, g4, tau_max, path):
    # 2D readable representation: capability lambda_max vs azimuth angle,
    # sampled in the x-y plane and in a tilted plane, for both geometries.
    n = 720
    angles = np.linspace(0, 2 * np.pi, n)

    def envelope_in_plane(g, plane_vecs):
        e1, e2 = plane_vecs
        dirs = np.array([np.cos(a) * e1 + np.sin(a) * e2 for a in angles])
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        return directional_capability_envelope(g, tau_max, dirs)

    xy_plane = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    tilt_plane = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2), np.array([0.0, 0.0, 1.0]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})

    for ax, plane, title in zip(axs, (xy_plane, tilt_plane), ("x-y plane slice", "(x+y)/z tilted-plane slice")):
        env3 = envelope_in_plane(g3, plane)
        env4 = envelope_in_plane(g4, plane)
        ax.plot(angles, env3, label="3-wheel orthogonal", color="tab:blue")
        ax.plot(angles, env4, label="4-wheel tetrahedral", color="tab:orange")
        ax.set_title(title)
        ax.set_rlabel_position(135)

    axs[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.suptitle(r"Figure 3 — Directional Body-Torque Capability $\lambda_{max}(\hat u)$ [N·m]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure4_failure_capability(g4, failure_results, iso4_nominal, tau_max, path):
    n = 720
    angles = np.linspace(0, 2 * np.pi, n)

    def envelope_in_plane(g, plane_vecs):
        e1, e2 = plane_vecs
        dirs = np.array([np.cos(a) * e1 + np.sin(a) * e2 for a in angles])
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        return directional_capability_envelope(g, tau_max, dirs)

    # Two slice planes: the x-y slice alone makes wheel-failure pairs
    # (W1<->W4, W2<->W3 here) coincide EXACTLY by a genuine reflection
    # symmetry of the tetrahedral geometry -- verified numerically, not a
    # plotting bug. The second, tilted plane breaks that degeneracy so
    # all four single-failure cases are visually distinguishable.
    xy_plane = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    tilt_plane = (np.array([1.0, 1.0, 0.0]) / np.sqrt(2), np.array([0.0, 0.0, 1.0]))

    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    linestyles = ["-", "--", "-.", ":"]

    fig, axs = plt.subplots(1, 2, figsize=(13, 7), subplot_kw={"projection": "polar"})
    for ax, plane, title in zip(axs, (xy_plane, tilt_plane), ("x-y plane slice", "(x+y)/z tilted-plane slice")):
        env_nominal = envelope_in_plane(g4, plane)
        ax.plot(angles, env_nominal, color="black", linewidth=2.5, label="Nominal (4 wheels)")
        for idx in range(4):
            g_failed = failure_results[idx]["geometry"]
            env_failed = envelope_in_plane(g_failed, plane)
            ax.plot(angles, env_failed, color=colors[idx], linestyle=linestyles[idx],
                    alpha=0.85, linewidth=1.8, label=f"Wheel {g4.labels[idx]} failed")
        ax.set_title(title)

    axs[0].legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)
    fig.suptitle(
        r"Figure 4 — Nominal vs Single-Wheel-Failure Directional Capability [N·m]"
        "\n(x-y slice: failure pairs coincide by reflection symmetry -- see tilted slice for all 4)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_table(table_rows, wheel, path):
    lines = []
    lines.append("# Milestone 2 — Wheel Loading Table\n")
    lines.append(
        f"Every wheel shares the representative capability: tau_max = {wheel.tau_max} N*m, "
        f"H_max = {wheel.H_max:.4f} N*m*s (J_w = {wheel.J_w} kg*m^2).\n"
    )
    lines.append(
        "| Case | Geometry | Body torque τ_c [N·m] | Wheel torque τ_w [N·m] | max τ_w | "
        "ρ_τ,max | Worst wheel (τ) | Body momentum ΔH_c [N·m·s] | Wheel momentum h_w [N·m·s] | "
        "max h_w | ρ_H,max | Worst wheel (H) | Feasible? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in table_rows:
        tau_c_str = "[" + ", ".join(f"{v:.4f}" for v in r["tau_c"]) + "]"
        tau_w_str = "[" + ", ".join(f"{v:.4f}" for v in r["tau_w"]) + "]"
        dH_c_str = "[" + ", ".join(f"{v:.4f}" for v in r["dH_c"]) + "]"
        H_w_str = "[" + ", ".join(f"{v:.4f}" for v in r["H_w"]) + "]"
        lines.append(
            f"| {r['case']} | {r['geometry']} | {tau_c_str} | {tau_w_str} | "
            f"{np.max(np.abs(r['tau_w'])):.4f} | {r['rho_tau_max']:.3f} | {r['worst_tau_wheel']} | "
            f"{dH_c_str} | {H_w_str} | {np.max(np.abs(r['H_w'])):.4f} | {r['rho_H_max']:.3f} | "
            f"{r['worst_H_wheel']} | {'yes' if r['feasible'] else '**no**'} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

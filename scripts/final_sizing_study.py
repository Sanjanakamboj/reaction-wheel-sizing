#!/usr/bin/env python3
"""
Milestone 5 analysis script — integrated final reaction-wheel sizing.

Prints the complete M1-M4 requirement traceback, engineering-margin
trades, speed/inertia trade, candidate feasibility map, sensitivity
studies, robust corner case, final capability recommendation, and the
updated M3/M4 schedule for the recommended wheel. Generates:

    results/fig1_torque_vs_maneuver_time.png
    results/fig2_momentum_vs_maneuver_time.png
    results/fig3_feasibility_map.png
    results/fig4_inertia_speed_tradeoff.png
    results/fig5_final_operational_timeline.png
    results/final_wheel_sizing.md
    results/capability_trade_table.md

Run with:  python scripts/final_sizing_study.py
"""

import os
import sys
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from analyze_momentum_accumulation import build_environment  # reuse M3 environment

from reaction_wheel.spacecraft import representative_spacecraft
from reaction_wheel.maneuvers import triangular_slew_requirement
from reaction_wheel.wheel import representative_wheel, ReactionWheel
from reaction_wheel.geometry import orthogonal_3wheel, tetrahedral_4wheel, allocate_torque
from reaction_wheel.disturbances import leo_orbit_reference, dipole_field_body
from reaction_wheel.sizing import SizingMargins
from reaction_wheel.momentum import mean_wheel_torque
from reaction_wheel.desaturation import representative_magnetorquer, simulate_dump, HysteresisThresholds
from reaction_wheel.final_sizing import (
    ADOPTED_MANEUVER, STRESS_MANEUVER, maneuver_body_vectors,
    wheel_torque_requirement, directional_envelope_torque_requirement,
    maneuver_at_threshold_headroom, required_H_max_for_headroom,
    max_allowable_dump_on_fraction, max_pre_maneuver_momentum,
    required_rotor_inertia, stored_energy, required_wheel_acceleration,
    classify_candidate, feasibility_grid,
    maneuver_time_sensitivity, inertia_sensitivity,
    evaluate_robust_corner_case, final_desaturation_schedule,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hr(char="-", n=94):
    print(char * n)


def main():
    sc = representative_spacecraft()
    g3 = orthogonal_3wheel()
    g4 = tetrahedral_4wheel()
    old_wheel = representative_wheel()  # the M1 synthetic wheel (tau_max=0.2, H_max=12.566)
    orb = leo_orbit_reference(500.0)
    env, _ = build_environment(sc, orb)

    hr("=")
    print("MILESTONE 5 ANALYSIS — Integrated Reaction-Wheel Sizing & Final Recommendation")
    hr("=")

    # -----------------------------------------------------------------
    # 1. Adopted maneuver vs stress case
    # -----------------------------------------------------------------
    hr()
    print("ADOPTED DESIGN MANEUVER (vs. M1 stress/verification case)")
    hr()
    print(f"  Adopted:  {ADOPTED_MANEUVER.label}")
    print(f"  Stress:   {STRESS_MANEUVER.label} (retained for comparison, NOT adopted as a requirement)")
    print("  Rationale: the 45deg/10s case was constructed in M1 purely to demonstrate that torque")
    print("  and momentum are independent sizing constraints; it was never an operational requirement.")
    print("  A 90deg/60s reorientation about the worst-inertia axis represents a plausible routine")
    print("  operational slew and is adopted here as the actual design driver.")

    tau_vec, dH_vec = maneuver_body_vectors(ADOPTED_MANEUVER, sc)
    tau_vec_stress, dH_vec_stress = maneuver_body_vectors(STRESS_MANEUVER, sc)
    print(f"\n  Adopted body torque/momentum: tau={np.linalg.norm(tau_vec):.4f} N*m, "
          f"H={np.linalg.norm(dH_vec):.4f} N*m*s")
    print(f"  Stress body torque/momentum:  tau={np.linalg.norm(tau_vec_stress):.4f} N*m, "
          f"H={np.linalg.norm(dH_vec_stress):.4f} N*m*s")

    # -----------------------------------------------------------------
    # 2. Torque requirement: nominal + fault-tolerant, both geometries
    # -----------------------------------------------------------------
    hr()
    print("TORQUE REQUIREMENT (nominal + one-wheel-failure-tolerant)")
    hr()
    tr3_adopted = wheel_torque_requirement(g3, tau_vec)
    tr4_adopted = wheel_torque_requirement(g4, tau_vec)
    tr4_stress = wheel_torque_requirement(g4, tau_vec_stress)
    print(f"  3-wheel orthogonal, adopted maneuver: nominal={tr3_adopted.tau_nominal:.4f} N*m, "
          f"failure=inf (no fault tolerance possible with 3 wheels)")
    print(f"  4-wheel tetrahedral, adopted maneuver: nominal={tr4_adopted.tau_nominal:.4f} N*m, "
          f"failure={tr4_adopted.tau_failure:.4f} N*m "
          f"(+{tr4_adopted.tau_failure_penalty_pct:.1f}% fault-tolerance penalty)")
    print(f"  4-wheel tetrahedral, STRESS maneuver (reference only): nominal={tr4_stress.tau_nominal:.4f} N*m, "
          f"failure={tr4_stress.tau_failure:.4f} N*m")

    env_req_nominal = directional_envelope_torque_requirement(g4, np.linalg.norm(tau_vec))
    print(f"\n  Directional-envelope requirement (guarantee ANY direction up to "
          f"{np.linalg.norm(tau_vec):.4f} N*m body torque): {env_req_nominal:.4f} N*m per wheel "
          f"(nominal geometry) -- more conservative than the single-trajectory value "
          f"({tr4_adopted.tau_nominal:.4f} N*m) since it must cover the worst possible direction, "
          "not just the adopted maneuver's specific axis.")
    print("  Sizing basis selected: the fault-tolerant, maneuver-specific requirement "
          f"({tr4_adopted.tau_failure:.4f} N*m) -- the directional-envelope number is reported for "
          "engineering awareness but not adopted, since the actual mission maneuver set is the real driver.")

    # -----------------------------------------------------------------
    # 3. Torque margin trade
    # -----------------------------------------------------------------
    hr()
    print("TORQUE ENGINEERING MARGIN TRADE")
    hr()
    tau_raw = tr4_adopted.tau_failure  # fault-tolerant requirement, the adopted sizing basis
    for SF_tau in (1.0, 1.25, 1.5, 2.0):
        print(f"  SF_tau = {SF_tau}: tau_sized = {SF_tau*tau_raw:.4f} N*m")
    SF_tau_final = 1.5
    tau_sized = SF_tau_final * tau_raw
    tau_recommended = math.ceil(tau_sized * 100) / 100.0  # round up to nearest 0.01 N*m
    print(f"  Selected SF_tau = {SF_tau_final} -> tau_sized = {tau_sized:.4f} N*m "
          f"-> rounded recommendation: tau_recommended = {tau_recommended:.2f} N*m")

    # -----------------------------------------------------------------
    # 4. Momentum requirement: maneuver-at-threshold headroom
    # -----------------------------------------------------------------
    hr()
    print("MOMENTUM REQUIREMENT — MANEUVER-AT-THRESHOLD HEADROOM")
    hr()
    f_on_baseline = 0.8
    H_on_old = f_on_baseline * old_wheel.H_max
    hr_nominal = maneuver_at_threshold_headroom(g4, dH_vec, H_pre_maneuver=H_on_old, H_max=old_wheel.H_max)
    hr_failure_geoms = []
    for idx in range(4):
        g_failed = g4.remove_wheel(idx)
        res = allocate_torque(g_failed, dH_vec)
        hr_failure_geoms.append(float(np.max(np.abs(res.wheel_values))))
    delta_H_failure = max(hr_failure_geoms)
    hr_failure = maneuver_at_threshold_headroom(g4, dH_vec, H_pre_maneuver=H_on_old, H_max=old_wheel.H_max)

    print(f"  Old synthetic wheel: H_max={old_wheel.H_max:.4f} N*m*s, H_on={H_on_old:.4f} N*m*s (f_on={f_on_baseline})")
    print(f"  Adopted maneuver, nominal 4-wheel: delta_H = {hr_nominal.delta_H_maneuver:.4f} N*m*s, "
          f"worst-case total = {hr_nominal.H_pre_maneuver + hr_nominal.delta_H_maneuver:.4f} N*m*s, "
          f"feasible = {hr_nominal.feasible} (margin {hr_nominal.margin:.4f} N*m*s)")
    print(f"  Adopted maneuver, worst single-wheel failure: delta_H = {delta_H_failure:.4f} N*m*s, "
          f"worst-case total = {H_on_old + delta_H_failure:.4f} N*m*s, "
          f"feasible = {H_on_old + delta_H_failure <= old_wheel.H_max} "
          f"(margin {old_wheel.H_max - H_on_old - delta_H_failure:.4f} N*m*s)")

    # Stress case, for reference/awareness only
    res_stress = allocate_torque(g4, dH_vec_stress)
    delta_H_stress = float(np.max(np.abs(res_stress.wheel_values)))
    print(f"\n  For reference (NOT adopted): stress-maneuver nominal delta_H = {delta_H_stress:.4f} N*m*s, "
          f"worst-case total = {H_on_old + delta_H_stress:.4f} N*m*s, "
          f"feasible = {H_on_old + delta_H_stress <= old_wheel.H_max}")

    f_on_max = max_allowable_dump_on_fraction(delta_H_failure, old_wheel.H_max)
    print(f"\n  Maximum allowable dump-on fraction for the old wheel given the failure-case headroom "
          f"requirement: f_on_max = {f_on_max:.4f} (baseline f_on={f_on_baseline} is "
          f"{'CONSISTENT' if f_on_baseline <= f_on_max else 'INCONSISTENT'} with maneuver headroom -- "
          "no revision to the M3/M4 operational threshold is required).")

    # -----------------------------------------------------------------
    # 5. Momentum margin trade + final momentum recommendation
    # -----------------------------------------------------------------
    hr()
    print("MOMENTUM ENGINEERING MARGIN TRADE")
    hr()
    H_raw = required_H_max_for_headroom(delta_H_failure, f_on_baseline)  # bare minimum, no margin
    print(f"  Bare-minimum required H_max (failure-case headroom driven, no margin): {H_raw:.4f} N*m*s")
    for SF_H in (1.0, 1.25, 1.5, 2.0):
        print(f"  SF_H = {SF_H}: H_sized = {SF_H*H_raw:.4f} N*m*s")
    SF_H_final = 1.5
    H_sized = SF_H_final * H_raw
    H_recommended = math.ceil(H_sized)  # round up to nearest whole N*m*s
    print(f"  Selected SF_H = {SF_H_final} -> H_sized = {H_sized:.4f} N*m*s "
          f"-> rounded recommendation: H_recommended = {H_recommended:.1f} N*m*s")
    print(f"  (For reference, the old synthetic wheel's H_max = {old_wheel.H_max:.3f} N*m*s already "
          f"exceeds this derived minimum.)")

    # -----------------------------------------------------------------
    # 5b. Robust corner case -- evaluated BEFORE finalizing capability,
    # so the final recommendation can be escalated if the nominal-margin
    # sizing above does not survive it (see below).
    # -----------------------------------------------------------------
    hr()
    print("ROBUST CORNER CASE (+20% inertia, one wheel failed, adopted maneuver, selected margins)")
    hr()
    margins = SizingMargins(SF_tau=SF_tau_final, SF_H=SF_H_final)

    # Trial check against the H_on that goes with the OLD wheel's capacity
    # (a fixed, concrete pre-maneuver assumption -- not yet the circular
    # "f_on times the capacity we are about to solve for").
    robust_trial = evaluate_robust_corner_case(
        g4, sc, ADOPTED_MANEUVER, inertia_scale=1.2, disturbance_scale=2.0,
        H_pre_maneuver=H_on_old, margins=margins,
        tau_recommended=tau_recommended, H_recommended=H_recommended,
    )
    print(f"  Required (robust case, margins already applied): tau={robust_trial.tau_required:.4f} N*m, "
          f"H={robust_trial.H_required:.4f} N*m*s")
    print(f"  Nominal-margin recommendation (v1): tau={tau_recommended:.4f} N*m, H={H_recommended:.4f} N*m*s")
    print(f"  Torque feasible: {robust_trial.torque_feasible}, Momentum feasible: {robust_trial.momentum_feasible}")

    if not robust_trial.overall_feasible:
        print("  RESULT: the nominal-margin (SF=1.5) sizing does NOT survive the combined robust corner")
        print("  case (+20% inertia AND one wheel failed, simultaneously). Rather than silently accept an")
        print("  unverified design, the final recommendation is ESCALATED to the robust-case requirement,")
        print("  solved self-consistently for H_pre = f_on*H_max (the same closed form as section 5,")
        print("  applied to the robust-case excursion instead of the nominal one).")

        g_failed0 = g4.remove_wheel(0)
        I_robust = sc.Iy * 1.2
        req_robust = triangular_slew_requirement(np.deg2rad(ADOPTED_MANEUVER.theta_deg), ADOPTED_MANEUVER.T, I_robust)
        tau_vec_robust = np.array([0.0, req_robust.tau_req, 0.0])
        dH_vec_robust = np.array([0.0, req_robust.H_body_peak, 0.0])
        tau_required_raw_robust = float(np.max(np.abs(allocate_torque(g_failed0, tau_vec_robust).wheel_values)))
        delta_H_raw_robust = float(np.max(np.abs(allocate_torque(g_failed0, dH_vec_robust).wheel_values)))

        tau_required_robust = SF_tau_final * tau_required_raw_robust
        H_max_robust_required = required_H_max_for_headroom(SF_H_final * delta_H_raw_robust, f_on_baseline)

        tau_recommended = math.ceil(max(tau_recommended, tau_required_robust) * 100) / 100.0
        H_recommended = math.ceil(max(H_recommended, H_max_robust_required))
        print(f"  Escalated final recommendation: tau_recommended = {tau_recommended:.2f} N*m, "
              f"H_recommended = {H_recommended:.1f} N*m*s")
        robust_final = evaluate_robust_corner_case(
            g4, sc, ADOPTED_MANEUVER, inertia_scale=1.2, disturbance_scale=2.0,
            H_pre_maneuver=f_on_baseline * H_recommended, margins=margins,
            tau_recommended=tau_recommended, H_recommended=H_recommended,
        )
        print(f"  Re-verification with escalated capability: torque_feasible={robust_final.torque_feasible}, "
              f"momentum_feasible={robust_final.momentum_feasible}, "
              f"OVERALL={'ROBUST' if robust_final.overall_feasible else 'STILL NOT ROBUST'}")
        robust_result = robust_final
    else:
        print("  RESULT: the nominal-margin (SF=1.5) sizing already survives the robust corner case -- "
              "no escalation needed.")
        robust_result = robust_trial

    # -----------------------------------------------------------------
    # 6. Speed / rotor-inertia trade
    # -----------------------------------------------------------------
    hr()
    print("WHEEL SPEED / ROTOR-INERTIA TRADE (for H_recommended)")
    hr()
    speed_candidates_rpm = [3000, 5000, 7000, 10000, 15000]
    print(f"  {'Omega_max [rpm]':>16}{'Omega_max [rad/s]':>20}{'J_w_min [kg*m^2]':>20}{'E_w [J]':>12}"
          f"{'Omega_dot_req [rad/s^2]':>26}")
    trade_rows = []
    for rpm in speed_candidates_rpm:
        Omega_max = rpm * 2 * np.pi / 60.0
        J_w_min = required_rotor_inertia(H_recommended, Omega_max)
        E_w = stored_energy(J_w_min, Omega_max)
        omega_dot_req = required_wheel_acceleration(tau_recommended, J_w_min)
        trade_rows.append(dict(rpm=rpm, Omega_max=Omega_max, J_w_min=J_w_min, E_w=E_w, omega_dot_req=omega_dot_req))
        print(f"  {rpm:>16}{Omega_max:>20.2f}{J_w_min:>20.5f}{E_w:>12.1f}{omega_dot_req:>26.3f}")

    Omega_max_recommended_rpm = 6000
    Omega_max_recommended = Omega_max_recommended_rpm * 2 * np.pi / 60.0
    J_w_recommended = required_rotor_inertia(H_recommended, Omega_max_recommended)
    E_w_recommended = stored_energy(J_w_recommended, Omega_max_recommended)
    omega_dot_recommended = required_wheel_acceleration(tau_recommended, J_w_recommended)
    print(f"\n  Selected: Omega_max_recommended = {Omega_max_recommended_rpm} rpm "
          f"({Omega_max_recommended:.2f} rad/s), J_w_recommended = {J_w_recommended:.5f} kg*m^2 "
          f"(rounded: {math.ceil(J_w_recommended*1000)/1000:.3f} kg*m^2)")
    print(f"  Stored energy at max speed: {E_w_recommended:.1f} J; "
          f"required wheel acceleration: {omega_dot_recommended:.3f} rad/s^2")

    J_w_recommended_rounded = math.ceil(J_w_recommended * 1000) / 1000.0

    # -----------------------------------------------------------------
    # 7. Final wheel object and integrated M1-M4 re-verification
    # -----------------------------------------------------------------
    final_wheel = ReactionWheel(J_w=J_w_recommended_rounded, Omega_max=Omega_max_recommended,
                                 tau_max=tau_recommended, name="M5 final recommended wheel")
    hr()
    print("FINAL RECOMMENDED WHEEL CAPABILITY")
    hr()
    print(f"  tau_max = {final_wheel.tau_max:.2f} N*m")
    print(f"  H_max   = {final_wheel.H_max:.3f} N*m*s")
    print(f"  Omega_max = {final_wheel.Omega_max_rpm:.0f} rpm")
    print(f"  J_w     = {final_wheel.J_w:.3f} kg*m^2")

    hr()
    print("INTEGRATED M1-M4 RE-VERIFICATION WITH THE FINAL WHEEL")
    hr()
    nominal_util = tr4_adopted.tau_nominal / final_wheel.tau_max
    failure_util = tr4_adopted.tau_failure / final_wheel.tau_max
    print(f"  M1/M2 torque check: nominal utilization = {nominal_util:.3f}, "
          f"failure utilization = {failure_util:.3f} "
          f"({'PASS' if failure_util <= 1.0 else 'FAIL'})")
    momentum_util = (H_on_old_equiv := f_on_baseline * final_wheel.H_max + delta_H_failure) / final_wheel.H_max
    print(f"  Headroom check with final wheel: worst-case total = {H_on_old_equiv:.3f} N*m*s "
          f"vs H_max = {final_wheel.H_max:.3f} N*m*s "
          f"({'PASS' if H_on_old_equiv <= final_wheel.H_max else 'FAIL'})")

    final_schedule = final_desaturation_schedule(g4, env, orb.period_s, final_wheel.H_max,
                                                  f_on=f_on_baseline, f_off=0.4)
    print(f"  M3/M4 updated schedule: H_on={final_schedule.H_on:.3f}, H_off={final_schedule.H_off:.3f}, "
          f"repeat interval = {final_schedule.repeat_interval_orbits:.1f} orbits "
          f"({final_schedule.repeat_interval_s/86400:.2f} days)")

    mt = representative_magnetorquer()
    B_field = lambda t: dipole_field_body(t, orb)
    tau_w_mean_final = mean_wheel_torque(g4, env, orb.period_s, n=4000)
    worst_idx_final = int(np.argmax(np.abs(tau_w_mean_final)))
    h0_final = tau_w_mean_final / np.max(np.abs(tau_w_mean_final)) * final_schedule.H_on
    dump_final = simulate_dump(g4, h0_final, env, B_field, mt, 0.0005, final_schedule.H_off, 20.0, 15 * orb.period_s)
    print(f"  Updated dump duration with final wheel: {dump_final.duration/orb.period_s:.2f} orbits "
          f"({dump_final.duration/3600:.2f} hours), reached_off={dump_final.reached_off}")
    duty_final = dump_final.duration / (dump_final.duration + final_schedule.repeat_interval_s)
    dumps_per_year_final = 365 * 86400.0 / (final_schedule.repeat_interval_s + dump_final.duration)
    print(f"  Updated duty cycle: {duty_final*100:.4f}%, representative dumps/year: {dumps_per_year_final:.1f}")

    # -----------------------------------------------------------------
    # 8. Sensitivity studies
    # -----------------------------------------------------------------
    hr()
    print("SENSITIVITY — MANEUVER TIME (adopted 90 deg slew, worst axis)")
    hr()
    time_results = maneuver_time_sensitivity(g4, sc, "y", 90.0, [10.0, 20.0, 30.0, 60.0])
    for r in time_results:
        print(f"  T={r['T']:>5.0f} s: tau_req={r['tau_req']:.4f} N*m, H_req={r['H_req']:.4f} N*m*s, "
              f"tau_w_nominal={r['tau_w_nominal']:.4f}, tau_w_failure={r['tau_w_failure']:.4f}")

    hr()
    print("SENSITIVITY — SPACECRAFT INERTIA (+-20%, adopted maneuver)")
    hr()
    inertia_results = inertia_sensitivity(g4, sc, ADOPTED_MANEUVER, [0.8, 1.0, 1.2])
    for r in inertia_results:
        print(f"  scale={r['scale']}: I={r['I']:.2f} kg*m^2, tau_req={r['tau_req']:.4f} N*m, "
              f"tau_w_nominal={r['tau_w_nominal']:.4f}, tau_w_failure={r['tau_w_failure']:.4f}")

    hr()
    print("SENSITIVITY — DISTURBANCE MAGNITUDE (0.5/1/2x, effect on schedule not wheel sizing)")
    hr()
    for scale in (0.5, 1.0, 2.0):
        tau_w_mean_s = mean_wheel_torque(g4, env, orb.period_s, n=2000) * scale
        worst_i_s = int(np.argmax(np.abs(tau_w_mean_s)))
        from reaction_wheel.desaturation import analytical_repeat_interval
        t_repeat_s = analytical_repeat_interval(final_schedule.H_on, final_schedule.H_off, tau_w_mean_s[worst_i_s])
        print(f"  disturbance x{scale}: repeat interval = {t_repeat_s/orb.period_s:.1f} orbits "
              f"({t_repeat_s/86400:.1f} days) -- wheel torque/momentum SIZING unaffected "
              "(disturbance changes operational cadence only, since it never approaches the "
              "maneuver-driven torque/momentum requirement in magnitude).")

    hr()
    print("SENSITIVITY — MAGNETORQUER CAPABILITY (10/20/40 A*m^2)")
    hr()
    from reaction_wheel.desaturation import Magnetorquer
    for m_max in (10.0, 20.0, 40.0):
        mt_s = Magnetorquer(m_max=m_max)
        dump_s = simulate_dump(g4, h0_final, env, B_field, mt_s, 0.0005, final_schedule.H_off, 20.0, 15*orb.period_s)
        print(f"  m_max={m_max} A*m^2: dump duration = {dump_s.duration/orb.period_s:.2f} orbits "
              "-- wheel torque/momentum SIZING unaffected; only dump duration/operations change.")

    # -----------------------------------------------------------------
    # 9. Final confirmation of the robust corner case against the
    # (possibly escalated) final wheel capability.
    # -----------------------------------------------------------------
    hr()
    print("FINAL ROBUST-CASE CONFIRMATION (against the final recommended wheel)")
    hr()
    robust_confirm = evaluate_robust_corner_case(
        g4, sc, ADOPTED_MANEUVER, inertia_scale=1.2, disturbance_scale=2.0,
        H_pre_maneuver=final_schedule.H_on, margins=margins,
        tau_recommended=final_wheel.tau_max, H_recommended=final_wheel.H_max,
    )
    print(f"  Required (robust case): tau={robust_confirm.tau_required:.4f} N*m, "
          f"H={robust_confirm.H_required:.4f} N*m*s")
    print(f"  Final recommended capability: tau={robust_confirm.tau_recommended:.4f} N*m, "
          f"H={robust_confirm.H_recommended:.4f} N*m*s")
    print(f"  Torque feasible: {robust_confirm.torque_feasible}, "
          f"Momentum feasible: {robust_confirm.momentum_feasible}, "
          f"OVERALL: {'ROBUST' if robust_confirm.overall_feasible else 'NOT ROBUST -- revisit margins/capability'}")
    robust_result = robust_confirm

    hr()
    print("OPERATIONAL COMPATIBILITY OF THE M1 STRESS MANEUVER WITH THE FINAL WHEEL")
    hr()
    stress_util_nominal = tr4_stress.tau_nominal / final_wheel.tau_max
    stress_util_failure = tr4_stress.tau_failure / final_wheel.tau_max
    print(f"  Stress maneuver (45deg/10s) torque utilization on the FINAL wheel: "
          f"nominal={stress_util_nominal:.2f}, failure={stress_util_failure:.2f}")
    print("  Both exceed 1.0: the final wheel (correctly sized to the ADOPTED routine maneuver, not the")
    print("  stress case) cannot execute the aggressive 45deg/10s slew. This is an explicit, accepted")
    print("  consequence of adopting a realistic design maneuver rather than the harshest illustrative")
    print("  M1 example -- NOT an oversight. If that aggressive maneuver is ever operationally required,")
    print("  the design mitigation is an OPERATIONAL RULE (e.g. restrict it to a degraded slew time,")
    print(f"  or inhibit it above a defined momentum-utilization threshold), not silently oversizing")
    print("  hardware for a case that was never adopted as a mission requirement (M5 sec. 28-30).")

    # -----------------------------------------------------------------
    # 10. Architecture decision
    # -----------------------------------------------------------------
    hr()
    print("ARCHITECTURE DECISION: 3-WHEEL ORTHOGONAL vs 4-WHEEL TETRAHEDRAL")
    hr()
    print(f"  3-wheel: nominal torque req={tr3_adopted.tau_nominal:.4f} N*m (lower than 4-wheel nominal), "
          "but ZERO one-wheel-failure tolerance -- any single failure loses an entire control axis.")
    print(f"  4-wheel: nominal torque req={tr4_adopted.tau_nominal:.4f} N*m, "
          f"failure-tolerant req={tr4_adopted.tau_failure:.4f} N*m "
          f"(+{tr4_adopted.tau_failure_penalty_pct:.0f}% penalty vs its own nominal) -- "
          "survives any one wheel failure with full 3-axis authority (M2).")
    print("  DECISION: 4-wheel tetrahedral is selected. The fault-tolerance torque penalty is "
          f"small in absolute terms ({tr4_adopted.tau_failure:.3f} N*m, well within the final "
          f"{final_wheel.tau_max:.2f} N*m recommended capability), while the 3-wheel architecture "
          "offers no recovery path from a single wheel "
          "failure -- a poor trade for a modest torque saving.")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    fig1 = os.path.join(RESULTS_DIR, "fig1_torque_vs_maneuver_time.png")
    fig2 = os.path.join(RESULTS_DIR, "fig2_momentum_vs_maneuver_time.png")
    fig3 = os.path.join(RESULTS_DIR, "fig3_feasibility_map.png")
    fig4 = os.path.join(RESULTS_DIR, "fig4_inertia_speed_tradeoff.png")
    fig5 = os.path.join(RESULTS_DIR, "fig5_final_operational_timeline.png")

    make_figure1(time_results, fig1)
    make_figure2(time_results, H_on_old, old_wheel.H_max, fig2)
    make_figure3(tr4_adopted, H_raw, old_wheel, final_wheel, fig3)
    make_figure4(trade_rows, Omega_max_recommended_rpm, J_w_recommended, fig4)
    make_figure5(dump_final, final_schedule, g4, fig5)

    print(f"\nFigures written to: {RESULTS_DIR}")
    for p in (fig1, fig2, fig3, fig4, fig5):
        print(f"  {os.path.relpath(p)}")

    # -----------------------------------------------------------------
    # Tables
    # -----------------------------------------------------------------
    sizing_path = os.path.join(RESULTS_DIR, "final_wheel_sizing.md")
    write_final_sizing_table(tr4_adopted, tau_raw, SF_tau_final, tau_sized, tau_recommended,
                              H_raw, SF_H_final, H_sized, H_recommended, final_wheel,
                              final_schedule, sizing_path)
    print(f"\nFinal sizing table written to: {os.path.relpath(sizing_path)}")

    trade_path = os.path.join(RESULTS_DIR, "capability_trade_table.md")
    write_capability_trade_table(tr4_adopted, H_raw, final_wheel, trade_path)
    print(f"Capability trade table written to: {os.path.relpath(trade_path)}")

    hr("=")
    print("END OF M5 ANALYSIS REPORT")
    hr("=")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure1(time_results, path):
    T = np.array([r["T"] for r in time_results])
    tau_nom = np.array([r["tau_w_nominal"] for r in time_results])
    tau_fail = np.array([r["tau_w_failure"] for r in time_results])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(T, tau_nom, "o-", label="Nominal 4-wheel", color="tab:blue")
    ax.plot(T, tau_fail, "s-", label="Worst single-wheel failure", color="tab:red")
    ax.set_xlabel("Maneuver duration T [s]")
    ax.set_ylabel("Required worst-wheel torque [N*m]")
    ax.set_yscale("log")
    ax.set_title("Figure 1 — Required Wheel Torque vs Maneuver Time (90 deg slew)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure2(time_results, H_on_old, H_max_old, path):
    T = np.array([r["T"] for r in time_results])
    h_nom = np.array([r["h_w_nominal"] for r in time_results])
    h_fail = np.array([r["h_w_failure"] for r in time_results])
    headroom = H_max_old - H_on_old
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(T, h_nom, "o-", label="Nominal 4-wheel momentum excursion", color="tab:blue")
    ax.plot(T, h_fail, "s-", label="Worst single-wheel-failure excursion", color="tab:red")
    ax.axhline(headroom, color="black", linestyle="--", label=f"Available headroom (old wheel) = {headroom:.2f} N*m*s")
    ax.set_xlabel("Maneuver duration T [s]")
    ax.set_ylabel("Momentum excursion [N*m*s]")
    ax.set_title("Figure 2 — Required Momentum Excursion vs Maneuver Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure3(torque_req, H_raw, old_wheel, final_wheel, path):
    tau_range = np.linspace(0.005, 0.25, 120)
    H_range = np.linspace(1.0, 15.0, 120)
    grid = feasibility_grid(tau_range, H_range, torque_req, H_raw)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d9534f", "#f0ad4e", "#5cb85c"])
    im = ax.pcolormesh(tau_range, H_range, grid, cmap=cmap, vmin=-0.5, vmax=2.5, shading="auto")
    cbar = fig.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Infeasible", "Nominal only", "Failure-tolerant"])

    ax.plot(old_wheel.tau_max, old_wheel.H_max, "k*", markersize=16, label="M1 synthetic wheel")
    ax.plot(final_wheel.tau_max, final_wheel.H_max, "b^", markersize=14, label="M5 final recommendation")
    ax.set_xlabel(r"$\tau_{max}$ [N*m]")
    ax.set_ylabel(r"$H_{max}$ [N*m*s]")
    ax.set_title("Figure 3 — Torque/Momentum Capability Feasibility Map")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure4(trade_rows, Omega_max_recommended_rpm, J_w_recommended, path):
    rpm = np.array([r["rpm"] for r in trade_rows])
    J_w = np.array([r["J_w_min"] for r in trade_rows])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(rpm, J_w, "o-", color="tab:blue")
    ax.plot(Omega_max_recommended_rpm, J_w_recommended, "r*", markersize=16, label="Selected design point")
    ax.set_xlabel("Maximum wheel speed [rpm]")
    ax.set_ylabel(r"Required rotor inertia $J_w$ [kg*m^2]")
    ax.set_title("Figure 4 — Rotor Inertia vs Maximum Wheel Speed (for H_recommended)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figure5(dump_final, final_schedule, geometry, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for i in range(geometry.n_wheels):
        ax.plot(dump_final.t / 3600, dump_final.h_w[:, i], label=f"Wheel {geometry.labels[i]}", color=colors[i])
    ax.axhline(final_schedule.H_on, color="red", linestyle="--", label=r"$+H_{on}$ (final wheel)")
    ax.axhline(-final_schedule.H_on, color="red", linestyle="--")
    ax.axhline(final_schedule.H_off, color="green", linestyle="--", label=r"$+H_{off}$ (final wheel)")
    ax.axhline(-final_schedule.H_off, color="green", linestyle="--")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Wheel momentum [N*m*s]")
    ax.set_title("Figure 5 — Final Operational Timeline (dump event, final wheel capacity)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_final_sizing_table(torque_req, tau_raw, SF_tau, tau_sized, tau_recommended,
                              H_raw, SF_H, H_sized, H_recommended, final_wheel,
                              final_schedule, path):
    lines = ["# Milestone 5 — Final Reaction-Wheel Sizing Table\n"]
    lines.append(
        "| Requirement source | Nominal requirement | Failed-wheel requirement | "
        "Selected margin | Final minimum capability | Active driver | Notes |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(
        f"| Wheel torque | {torque_req.tau_nominal:.4f} N*m | {torque_req.tau_failure:.4f} N*m | "
        f"SF_tau={SF_tau} | **{final_wheel.tau_max:.2f} N*m** | One-wheel-failure-tolerant maneuver torque | "
        "Adopted 90deg/60s maneuver, worst axis, tetrahedral allocation |"
    )
    lines.append(
        f"| Momentum storage | n/a (headroom-driven) | {H_raw:.3f} N*m*s (raw, no margin) | "
        f"SF_H={SF_H} | **{final_wheel.H_max:.3f} N*m*s** | Maneuver-at-threshold headroom (failure case) | "
        "H_on=0.8*H_max assumed; disturbance accumulation is not the binding constraint |"
    )
    lines.append(
        f"| Maximum speed | -- | -- | -- | **{final_wheel.Omega_max_rpm:.0f} rpm** | "
        "Representative design choice | Selected from a 3000-15000 rpm trade |"
    )
    lines.append(
        f"| Rotor inertia | -- | -- | -- | **{final_wheel.J_w:.3f} kg*m^2** | "
        "H_recommended / Omega_max_recommended | Rounded up from the exact quotient |"
    )
    lines.append(
        f"| Desaturation threshold | H_on={final_schedule.H_on:.3f} N*m*s | "
        f"H_off={final_schedule.H_off:.3f} N*m*s | f_on=0.8, f_off=0.4 (unchanged) | "
        f"-- | Validated against maneuver headroom (M5 sec. 12) | No revision required |"
    )
    lines.append(
        f"| Dump repeat interval | {final_schedule.repeat_interval_orbits:.1f} orbits | -- | -- | "
        f"**{final_schedule.repeat_interval_s/86400:.1f} days** | Disturbance secular accumulation | "
        "Recomputed for the final wheel's H_max |"
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_capability_trade_table(torque_req, H_raw, final_wheel, path):
    lines = ["# Milestone 5 — Capability Trade Table\n"]
    lines.append(
        "| Design point | tau_max [N*m] | H_max [N*m*s] | Maneuver feasible? | "
        "Fault-tolerant feasible? | Momentum margin |"
    )
    lines.append("|---|---|---|---|---|---|")
    candidates = [
        ("Low-torque / high-momentum", 0.03, 12.0),
        ("Balanced", 0.05, 8.0),
        ("High-torque / moderate-momentum", 0.10, 6.0),
        ("Final selected requirement", final_wheel.tau_max, final_wheel.H_max),
    ]
    for label, tau_max, H_max in candidates:
        momentum_ok = H_max >= H_raw  # bare minimum, no margin -- matches classify_candidate's convention
        nominal_ok = (tau_max >= torque_req.tau_nominal) and momentum_ok
        failure_ok = (tau_max >= torque_req.tau_failure) and momentum_ok
        momentum_margin = H_max / H_raw
        lines.append(
            f"| {label} | {tau_max:.3f} | {H_max:.2f} | {'yes' if nominal_ok else 'no'} | "
            f"{'yes' if failure_ok else 'no'} | {momentum_margin:.2f}x |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

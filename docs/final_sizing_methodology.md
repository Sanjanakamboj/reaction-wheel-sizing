# Milestone 5 — Final Reaction-Wheel Sizing Methodology

This document explains the requirement hierarchy, the adopted design
maneuver, the margin and headroom logic, the speed/rotor-inertia trade,
the sensitivity and robustness studies, and the final architecture and
capability recommendation implemented in
[`final_sizing.py`](../src/reaction_wheel/final_sizing.py). It composes
the full M1–M4 pipeline; it does not re-derive any of that pipeline's
underlying physics.

## 1. Requirement hierarchy

Four distinct wheel capabilities are sized, and are never conflated:

| Requirement | Formula | Driven by |
|---|---|---|
| Torque $\tau_{\rm wheel,req}$ | $\max_i |\tau_{w,i}|$ over required cases | Maneuver-driven (M1/M2) |
| Momentum $H_{\rm wheel,req}$ | $\max_i |h_{w,i}|$ over required cases | Maneuver + accumulation + headroom (M1/M3/M4) |
| Speed $\Omega_{\rm req}$ | $H_{\rm req}/J_w$ | Derived from momentum + chosen rotor inertia |
| Rotor inertia $J_{w,\rm req}$ | $H_{\rm req}/\Omega_{\max}$ | Derived from momentum + chosen speed limit |

## 2. Adopted design maneuver vs. the M1 stress case

M1's 45°/10 s "aggressive" maneuver was constructed **specifically to
demonstrate that torque and momentum are independent sizing constraints**
(see `docs/wheel_sizing_methodology.md`) — it was never an operational
requirement. Sizing the whole spacecraft to it would be sizing to an
illustrative artifact, not a mission need.

M5 adopts **90°/60 s about the worst-inertia axis** ($I_y$) as the actual
design maneuver — a plausible routine full-attitude reorientation (e.g.
between imaging and comms pointing) at a realistic duration. The 45°/10 s
case is retained throughout M5 for comparison (`STRESS_MANEUVER` in
`final_sizing.py`) but never adopted as a requirement, and its
incompatibility with the final wheel is reported explicitly (§9) rather
than hidden.

## 3. Torque requirement: nominal, fault-tolerant, and directional

For the adopted maneuver's body torque vector, `wheel_torque_requirement`
computes the nominal (4-wheel) worst-wheel torque and, by iterating over
every single-wheel-removed subset (reusing `geometry.remove_wheel` and
`allocate_torque`), the worst-case one-wheel-failure requirement:

| | Nominal | Worst failure | Penalty |
|---|---|---|---|
| 4-wheel tetrahedral | 0.0212 N·m | 0.0423 N·m | **+100%** |

A 3-wheel orthogonal geometry has **no** failure-tolerant requirement at
all — removing any wheel leaves only 2, which cannot span 3-axis torque
(`WheelSetGeometry` itself enforces $N\ge3$); this is represented as
`tau_failure = inf`, not silently treated as equal to nominal.

A more general **directional-envelope** requirement — guaranteeing *any*
body-torque direction up to the adopted maneuver's magnitude, not just
its specific axis — is computed via `directional_envelope_torque_
requirement` (reusing M2's `isotropy_ratio` at unit capability, scaled
linearly): 0.0367 N·m, more conservative than the single-trajectory value
since it must cover the worst possible direction. **This more
conservative number is reported for engineering awareness but not
adopted as the sizing basis** — the mission's actual maneuver set (not an
arbitrary worst direction) is the real driver.

## 4. Momentum requirement: maneuver-at-threshold headroom

The physical momentum capacity must be large enough that the adopted
maneuver can complete **even if it begins right at the dump-on
threshold** $H_{\rm on}=f_{\rm on}H_{\max}$, without exceeding $H_{\max}$:

$$H_{\rm on} + \Delta H_{\rm maneuver} \le H_{\max}
\quad\Longrightarrow\quad
H_{\max} \ge \frac{\Delta H_{\rm maneuver}}{1-f_{\rm on}}$$

(`required_H_max_for_headroom`; the inverse, `max_allowable_dump_on_
fraction`, solves for the largest $f_{\rm on}$ instead). Using the
**worst single-wheel-failure** momentum excursion (the more demanding of
nominal vs. failure, per M5's conservative-case discipline),
$\Delta H_{\rm maneuver}=1.270$ N·m·s against the old wheel's $H_{\max}=
12.566$, $f_{\rm on}=0.8$: worst-case total $=11.323 \le 12.566$ N·m·s —
**feasible**, with 1.244 N·m·s of margin. The maximum allowable $f_{\rm
on}$ for this wheel is 0.899, so the M3/M4 baseline choice of 0.8 is
**consistent** with maneuver headroom — no revision to the operational
threshold was required by this analysis (a genuinely tested conclusion,
not a default retention).

## 5. Engineering margins

Torque and momentum margins ($SF_\tau$, $SF_H$) are evaluated at 1.0,
1.25, 1.5, 2.0 and a **1.5×** factor is selected for both — consistent
with the representative factor already used in M1's `sizing.
SizingMargins`, not a newly-invented number. Margin is applied to the
**raw excursion/requirement only**, never compounded onto a pre-existing
threshold that itself scales with the capacity being solved (see §7's
bug-fix discussion for why that distinction matters).

## 6. Speed / rotor-inertia trade

For the momentum-margined requirement, five representative (not
commercial-specific) maximum-speed candidates are traded off:

| $\Omega_{\max}$ [rpm] | $J_{w,\min}$ [kg·m²] | $E_w$ [J] | $\dot\Omega_{\rm req}$ [rad/s²] |
|---|---|---|---|
| 3000 | 0.0382 | 1885 | 2.09 |
| 5000 | 0.0229 | 3142 | 3.49 |
| 7000 | 0.0164 | 4398 | 4.89 |
| 10000 | 0.0115 | 6283 | 6.98 |
| 15000 | 0.0076 | 9425 | 10.47 |

6000 rpm is selected — a representative mid-range choice consistent with
common smallsat reaction-wheel speeds, giving $J_w=0.0191$ kg·m²
(rounded up to 0.020 kg·m² for the final recommendation).

## 7. The robust corner case, and a bug it caught

The deterministic robust corner case combines **+20% spacecraft inertia**
and **one wheel failed** with the adopted maneuver and selected margins
(`evaluate_robust_corner_case`). Checking the nominal-margin (SF=1.5)
recommendation against this combined case **failed** — a genuine,
reported result, not smoothed over:

> Required (robust case): τ=0.0762 N·m, H=12.34–17.4 N·m·s (see below)
> vs. nominal-margin recommendation: τ=0.07 N·m, H=10.0 N·m·s → **NOT ROBUST**

**A modeling bug was caught and fixed during this check.** The first
implementation of `evaluate_robust_corner_case` applied the momentum
margin to the *entire* $(H_{\rm pre}+\Delta H)$ sum:
$H_{\rm required}=SF_H\cdot(H_{\rm pre}+\Delta H)$. Since $H_{\rm pre}$ is
itself $f_{\rm on}H_{\max}$ — a quantity that scales with the very
capacity being solved for — this compounds margin onto the threshold
itself, and the self-consistency equation
$H_{\max} = SF_H(f_{\rm on}H_{\max}+\Delta H)$ has **no finite solution**
whenever $SF_H\cdot f_{\rm on}\ge1$ (exactly true here: $1.5\times0.8=1.2$).
Escalating $H_{\max}$ under this formula made the requirement grow
*faster* than the escalation, diverging instead of converging. The fix —
matching §4/§5's established convention exactly — applies margin only to
the excursion: $H_{\rm required} = H_{\rm pre} + SF_H\Delta H$, giving the
well-defined closed form $H_{\max}\ge SF_H\Delta H/(1-f_{\rm on})$ for any
$f_{\rm on}<1$. This is now a named regression test
(`test_robust_corner_case_margin_applies_only_to_excursion_not_
preexisting_threshold`).

With the fix, the escalated recommendation ($\tau=0.08$ N·m, $H=12.0\to
12.566$ N·m·s after $J_w$ rounding) **passes** the robust corner case
confirmation exactly.

## 7b. Concurrency: which conditions can actually combine

Before adopting a "worst case," each candidate combination is checked
for physical or operational plausibility rather than assembled
automatically:

| Combination | Classification | Included in the robust case? |
|---|---|---|
| Spacecraft inertia uncertainty (+20%) + one wheel failed | Physically simultaneous — a mass-property estimation error does not depend on wheel health | **Yes** |
| One wheel failed + adopted (routine) maneuver | Physically simultaneous — the spacecraft must still be able to slew after a failure | **Yes** |
| One wheel failed + adopted maneuver + inertia uncertainty | Physically simultaneous (all three are independent physical facts that can co-occur) | **Yes — this is the robust case** |
| One wheel failed + the **un-adopted stress maneuver** (45°/10 s) | Not a required case — the stress maneuver was never adopted as an operational requirement (§2), so it is not combined with anything for sizing purposes | **No** (reported separately, §10, as an explicit operational restriction rather than folded into hardware sizing) |
| Near dump-on threshold + doubled disturbance | Independent sizing/operations question — doubled disturbance changes *how often* the threshold is reached, not the *momentum excursion* once a maneuver starts there | Evaluated separately (§9); not combined with the inertia+failure robust case, since disturbance magnitude does not change the momentum-headroom arithmetic (§4) |
| Active desaturation (magnetorquer driving) + adopted maneuver | Operationally avoidable in practice (a real program can defer non-urgent maneuvers during an active dump), and the M4 desaturation torques are five orders of magnitude below maneuver torques (M4 §7) — combining them would not change the sizing conclusion | Not combined; noted as operationally negligible rather than a sizing driver |

The robust case therefore combines exactly the physically-simultaneous,
sizing-relevant factors (inertia uncertainty, wheel failure, the adopted
maneuver) and deliberately excludes the un-adopted stress maneuver and
the disturbance-magnitude sweep, each for a stated reason — not an
unconstrained "everything bad happens at once" scenario, which would
produce an artificially inflated, physically meaningless requirement.

## 8. Final recommendation

| Quantity | Value | Driven by |
|---|---|---|
| $\tau_{\max}$ | **0.08 N·m** | Robust corner case (one-wheel-failure + 20% inertia uncertainty), $SF_\tau=1.5$ |
| $H_{\max}$ | **12.566 N·m·s** | Robust corner case headroom, $SF_H=1.5$ (converges almost exactly to the original M1 synthetic wheel's capacity — see §10) |
| $\Omega_{\max}$ | **6000 rpm** | Representative design choice from the speed/inertia trade |
| $J_w$ | **0.020 kg·m²** | $H_{\max}/\Omega_{\max}$, rounded up |

Torque sizing is driven by **fault tolerance** (the one-wheel-failure
case, itself amplified by the robust corner case's inertia uncertainty);
momentum sizing is driven by **maneuver-at-threshold headroom**, not by
environmental disturbance accumulation — the representative disturbance
environment is weak enough that momentum sizing never approaches it in
magnitude (§9).

## 9. Sensitivity studies

- **Maneuver time** (10/20/30/60 s, fixed 90° angle): confirms
  $\tau_{\rm req}\propto T^{-2}$ and $H_{\rm req}\propto T^{-1}$ exactly
  (reusing M1's triangular-slew formula unchanged). At 10 s, the
  fault-tolerant torque requirement (1.52 N·m) would demand a much larger
  wheel than the adopted 60 s case — a direct visualization of why the
  adopted duration matters (Figure 1).
- **Spacecraft inertia** ($\pm20\%$): confirms $\tau_{\rm req}\propto I$
  linearly, as expected from $\tau=I\alpha$.
- **Disturbance magnitude** (0.5×/1×/2×): changes the repeat interval
  (779→390 orbits at 2×) but **never changes wheel torque/momentum
  sizing** — disturbance-driven accumulation is always far below the
  maneuver-driven requirement in magnitude for this representative
  environment. Disturbance severity is an *operations* (dump frequency)
  concern, not a *hardware* sizing concern, for this spacecraft.
- **Magnetorquer capability** (10/20/40 A·m²): changes dump *duration*
  (6.9→1.8 orbits) but has **zero effect on wheel sizing** — the
  magnetorquer and the reaction wheels are sized against completely
  independent requirements (external unloading authority vs. internal
  maneuver/storage capability).

## 10. Architecture decision and integrated re-verification

**4-wheel tetrahedral is selected.** The fault-tolerance torque penalty
is small in absolute terms (0.042 N·m, well within the final 0.08 N·m
capability) against the alternative of a 3-wheel architecture that offers
**zero** recovery path from any single wheel failure — an entire control
axis is lost outright. This is a real trade evaluated with actual M1–M4
numbers, not a default assumption that "four is more redundant."

Re-running the M1–M4 pipeline with the final wheel: nominal torque
utilization 0.265, failure utilization 0.529 (both feasible); headroom
check 11.32 ≤ 12.566 N·m·s (feasible); updated M3/M4 schedule — $H_{\rm
on}=10.053$, $H_{\rm off}=5.027$ N·m·s, repeat interval 779.3 orbits
(51.2 days), dump duration 3.59 orbits (5.66 h), duty cycle 0.459%,
~7.1 dumps/representative year. Because the final $H_{\max}$ converged to
almost exactly the M1 synthetic wheel's original value, this schedule is
numerically identical to M4's — a coincidence of the specific numbers
involved, not a hidden assumption (confirmed by recomputing it fresh from
the final wheel, not by copying M4's result forward).

**Operational note on the stress maneuver**: the final wheel's torque
utilization for the *un-adopted* 45°/10 s stress case is well above 1.0
in both nominal (4.76×) and failure (9.52×) modes. This is an accepted
consequence of sizing to the adopted routine maneuver, not an oversight.
If that aggressive maneuver is ever operationally required, the correct
mitigation is an **operational rule** (inhibit it above a defined
momentum-utilization threshold, or accept a degraded slew time in that
mode) — not silently oversizing hardware for a case that was never
adopted as a requirement.

## 11. Limitations

- The adopted maneuver (90°/60 s) is a representative engineering choice,
  not a specific mission requirement document.
- Margins ($SF_\tau=SF_H=1.5$) are representative, consistent with M1's
  convention, not independently derived from a specific verification or
  reliability policy.
- The robust corner case combines only the two factors most directly
  tied to wheel sizing (inertia uncertainty, wheel failure); it does not
  sweep every M1–M4 sensitivity simultaneously (that would require a
  much larger structured or Monte Carlo study, judged unnecessary for a
  systems-level actuator-sizing study per the M5 scope boundary).
- The directional-envelope requirement (§3) is reported but not adopted
  — a program with less confidence in its maneuver set might choose to
  adopt it instead, at the cost of a larger wheel.
- No commercial hardware comparison was performed (optional per the M5
  scope; the primary deliverable is the derived requirement, not a
  product selection).
- This remains a systems-level actuator sizing study: no rotor
  stress/FEM, bearing design, motor electromagnetic design, jitter,
  thermal, or power-system analysis.

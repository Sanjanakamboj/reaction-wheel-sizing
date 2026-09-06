# Milestone 3 — Disturbance Momentum Accumulation & Saturation-Time Methodology

This document explains the disturbance models, wheel-space allocation
reuse, momentum integration, and saturation-time analysis implemented in
[`disturbances.py`](../src/reaction_wheel/disturbances.py) and
[`momentum.py`](../src/reaction_wheel/momentum.py). It builds on, and does
not duplicate, the wheel mechanics of `wheel.py` and the geometry/
allocation math of `geometry.py`. It does **not** cover momentum
dumping/desaturation control — that is Milestone 4 scope; every number
here is a *time until unloading becomes necessary*, never a correction.

## 1. Disturbance-torque models and their parameter rationale

All disturbance components are **illustrative representative engineering
assumptions for a small LEO spacecraft**, not a mission-specific
environmental model, except where a value is a standard cited physical
constant. Every parameter used in the M3 baseline environment:

| Parameter | Value | Rationale |
|---|---|---|
| Solar pressure at ~1 AU | $4.56\times10^{-6}$ N/m² | Standard physical constant ($P=S/c$), not fitted |
| SRP reflectivity $C_R$ | 1.3 | Representative flat-plate coefficient |
| SRP/aero area $A$ | 2.0 m² | Representative smallsat cross-section |
| SRP center-of-pressure offset | 0.05 m along body z | Illustrative cg/cp offset |
| Atmospheric density $\rho$ (500 km) | $5\times10^{-13}$ kg/m³ | Representative solar-average LEO value; **strongly** altitude- and solar-activity-dependent — not claimed as mission-lifetime-accurate |
| Drag coefficient $C_D$ | 2.2 | Representative free-molecular-flow value |
| Aero center-of-pressure offset | 0.03 m along body y | Illustrative, chosen distinct from the SRP offset axis so the two secular biases load different wheels |
| Orbital altitude | 500 km circular | Representative LEO |
| Orbit inclination | 51.6° | Representative ISS-class inclination (shapes the magnetic-field direction model only) |
| Residual magnetic dipole $m_{\rm res}$ | $[0.05, 0.02, -0.03]$ A·m² | Representative small-sat residual dipole estimate |
| LEO field magnitude $B_0$ | $3\times10^{-5}$ T | Representative low/mid-latitude LEO magnitude (~30,000 nT) |

None of these values were tuned to force a convenient saturation time —
they were fixed from the rationale above before the saturation-time
results were computed (see §7 for the resulting, unforced numbers).

### Solar radiation pressure and aerodynamic drag

Both use the same simplified force-then-torque pattern:

$$F = P\,C\,A \text{ (SRP)}, \qquad F = \tfrac12\rho v^2 C_D A \text{ (drag)},
\qquad \tau = \mathbf r_{cp}\times \mathbf F.$$

Both are modeled here as **constant body-frame vectors** — i.e. the
sun/ram direction and center-of-pressure offset are held fixed in the
body frame. This is a deliberately simple, conservative "worst-case
secular bias" model, not a rotating-geometry simulation; a real
spacecraft's sun and ram vectors slowly change with orbit/attitude, which
would modulate these into slower oscillations rather than pure
constants — the qualitative secular-accumulation conclusion is not
sensitive to this simplification, only the exact numeric rate.

### Gravity-gradient torque

$$\boldsymbol\tau_{gg} = 3\frac{\mu}{r^3}\left(\hat r_b \times I\hat r_b\right)$$

Exactly zero when $\hat r_b$ is a principal inertia axis (verified in
`test_disturbances.py`). The M3 baseline models $\hat r_b(t)$ as sweeping
the body x-y plane once per orbit at the orbital mean motion — a
simplified stand-in for an inertially-pointed spacecraft's local-vertical
direction migrating through the body frame over one orbit. **This
produces an oscillation at *twice* the orbital rate** (visible in Figure
1): $\hat r_b\times I\hat r_b$ is quadratic in $\hat r_b$, so a
fundamental-frequency rotation of the local vertical produces a
second-harmonic torque — a real, not spurious, effect of the physics,
confirmed by inspection before being written up here (§9, visual
inspection).

### Residual magnetic-dipole torque

$$\boldsymbol\tau_m = \mathbf m_{\rm res}\times \mathbf B$$

`dipole_field_body(t)` is an explicitly simplified stand-in for a real
IGRF/tilted-dipole field history: it sweeps direction once per orbit with
an inclination-dependent tilt, giving the magnetic torque a physically
motivated orbital periodicity without claiming geomagnetic-model fidelity.

## 2. Secular vs. oscillatory disturbances — a key distinction

A disturbance with a nonzero time-mean **accumulates wheel momentum
approximately linearly**: $\Delta H \approx \tau_{\rm mean}\,t$. A
zero-mean periodic disturbance produces **bounded, non-secular** wheel
momentum — demonstrated exactly for a sinusoidal torque
$\tau_w(t)=\tau_0\sin(\omega t)$:

$$\Delta h_w(t) = \frac{\tau_0}{\omega}\left(1-\cos\omega t\right)
\quad\in\ \left[0,\ \tfrac{2\tau_0}{\omega}\right]\ \text{for all } t,$$

verified against the numerical integrator to $<10^{-6}$ N·m·s
(`test_momentum.py::test_integration_matches_analytical_sinusoidal_single_axis`)
and shown to stay bounded over 50 full periods
(`test_sinusoidal_momentum_is_bounded_not_secular`).

**Peak torque alone does not determine which disturbance drives long-term
accumulation.** In the M3 baseline environment, gravity-gradient has the
*largest peak* torque ($1.84\times10^{-5}$ N·m) but a *negligible* orbital
mean ($\sim2\times10^{-21}$ N·m — numerically zero, exactly as expected
for a torque that is an odd/quadratic function averaging out over a full
sweep), while aerodynamic drag has a much smaller peak
($1.91\times10^{-6}$ N·m) but is the dominant **secular** contributor
because it is modeled as constant. See the momentum-budget table
([`results/momentum_budget_table.md`](../results/momentum_budget_table.md))
for the full per-component breakdown.

## 3. Wheel-space disturbance allocation (reused, not duplicated)

Disturbance allocation reuses `geometry.allocate_torque` directly:

$$\boldsymbol\tau_w(t) = -A^{+}\boldsymbol\tau_d(t)$$

i.e. the disturbance vector is fed through the *same* verified
minimum-norm pseudoinverse map used for maneuver torque commands in M2.
This is a deliberate bookkeeping choice, not a rigorous closed-loop
disturbance-rejection-controller derivation (a full derivation would
relate the wheel torque command to the *negative* of the disturbance via
an attitude-hold control law, which differs from this formula only by an
overall sign). **Every engineering conclusion in this milestone —
momentum-accumulation magnitude, worst-wheel identification,
threshold-crossing time, and every geometry/failure comparison — is
invariant to that overall sign choice**, because the disturbance pointing
directions themselves are illustrative assumptions, not rigorously derived
physical directions. `wheel_torque_history` (in `momentum.py`) never
silently clips an infeasible instantaneous allocation — it raises
`InfeasibleDisturbanceError` (checked automatically whenever a `tau_max`
is supplied; for realistic disturbance magnitudes this does not trigger,
verified rather than assumed — the peak total disturbance torque here is
$\sim2\times10^{-5}$ N·m, five orders of magnitude below the
representative wheel's 0.2 N·m capability).

## 4. Momentum integration

$$h_{w,i}(t) = h_{w,i}(0) + \int_0^t \tau_{w,i}(t')\,dt'$$

integrated via cumulative trapezoidal quadrature
(`integrate_wheel_momentum`), verified against two exact analytical
solutions: constant torque ($h(t)=h_0+\tau_w t$, residual
$<10^{-9}$ N·m·s over a 6000 s / 4000-sample integration) and sinusoidal
torque (§2, residual $<10^{-6}$ N·m·s). Body/wheel momentum consistency —
$A\,\Delta\mathbf h_w(t) \approx -\int_0^t\boldsymbol\tau_d(t')\,dt'$ — is
verified numerically to $<10^{-6}$ N·m·s for both the orthogonal and
tetrahedral geometries (`test_body_wheel_momentum_consistency_*`).

## 5. Operational momentum threshold

The physical wheel momentum limit is $H_{\max}=J_w\Omega_{\max}$ (M1). M3
introduces a **separate, smaller operational threshold**

$$H_{\rm threshold} = f_H\,H_{\max}, \qquad f_H = 0.8$$

reserving 20% headroom below the physical limit rather than operating to
100% of capacity. For the representative wheel ($H_{\max}=12.566$ N·m·s),
this gives $H_{\rm threshold}=10.053$ N·m·s. $f_H=0.8$ is a representative
engineering choice, not a derived or sourced requirement — a real program
would set it from wheel-speed-dependent friction/power margins, momentum
sensor accuracy, and control-authority reserve requirements. The physical
limit $H_{\max}$ and the operational threshold $H_{\rm threshold}$ are
kept as clearly separate quantities throughout the code and reports.

## 6. Saturation / threshold-crossing time

For a single wheel under **constant** torque, the exact closed-form time
to reach $\pm H_{\rm threshold}$ (correctly signed toward whichever bound
$\tau_w$ drives the wheel toward) is implemented in
`analytical_constant_torque_saturation_time`. For **arbitrary** disturbance
histories, `first_threshold_crossing` performs a direct numerical search
over an integrated momentum history and returns `reached=False` (never a
fabricated time) if the threshold is not crossed within the simulated
horizon.

Because the M3 baseline's periodic components have near-zero orbital
mean (§2), its **secular** behavior is dominated by the constant SRP/
aero/magnetic-mean torque. M3 therefore estimates long-horizon
saturation time via the **mean-torque approximation**
($\bar\tau = \frac1T\int_0^T\boldsymbol\tau_d\,dt$, `mean_wheel_torque`)
fed into the exact constant-torque formula, and *verifies* this estimate
directly against a full numerical integration out to the estimated
crossing time. For the nominal 4-wheel baseline, the numerical crossing
time (1558.9 orbits) matches the mean-torque analytical estimate (1558.5
orbits) to **0.03%** — the periodic terms are negligible relative to the
secular drift at the scale of hundreds of orbits, so the (much cheaper)
mean-torque estimate is used for the sensitivity/failure/bias sweeps,
with this direct numerical cross-check establishing when that
substitution is valid (it would NOT be valid if the periodic amplitude
were comparable to $H_{\rm threshold}$ within the timescale of interest —
that is not the case here, verified rather than assumed).

## 7. Headline results (baseline environment, nominal geometries)

*(Full numbers, including all four single-wheel-failure cases and the
sensitivity sweep, are in
[`results/configuration_comparison_table.md`](../results/configuration_comparison_table.md)
and the M3 script's printed report.)*

- Dominant **secular** driver: aerodynamic drag (orbital mean
  $1.91\times10^{-6}$ N·m).
- Dominant **peak-torque** driver: gravity-gradient
  ($1.84\times10^{-5}$ N·m) — a **different** component, exactly the
  distinction emphasized in §2.
- 3-wheel orthogonal: limiting wheel Wz, $t_{\rm sat}=926$ orbits (60.8
  days).
- 4-wheel tetrahedral (nominal): limiting wheel W2, $t_{\rm sat}=1559$
  orbits (102.4 days) — **1.68× longer** than the 3-wheel baseline for
  this specific disturbance direction and geometry (this ratio is
  disturbance-direction-dependent, not a general property of adding a
  4th wheel — see §8).
- 4-wheel, any one wheel failed: mean $t_{\rm sat}\approx999$ orbits, a
  **35.9% reduction** relative to the nominal 4-wheel case.

## 8. Geometry comparison is disturbance-direction-dependent

Unlike M2's isotropic torque-*capability* comparison (which is a property
of the geometry alone), momentum-accumulation *lifetime* depends on how a
specific, fixed disturbance direction happens to project onto each
wheel's axis. The four single-wheel-failure cases in M3 are **not**
equal to each other here (908, 895, 1300, 895 orbits) — unlike M2's
perfectly symmetric capability degradation — because the fixed baseline
disturbance vector breaks the tetrahedral geometry's rotational symmetry
(a disturbance aligned with a wheel that is later removed has a very
different effect than one that only lightly loaded that wheel to begin
with). This is a genuinely different conclusion from M2 and is reported
here quantitatively rather than assumed to follow the same pattern.

## 9. Visual inspection notes

Every M3 figure was inspected before being written up. Two findings
during inspection led to script changes (not analysis changes): (1) the
default 20-orbit plotting horizon for the wheel-momentum-history figure
showed no visible trend (20 orbits is <2% of the ~1559-orbit saturation
time) and was extended to 400 orbits so the secular drift toward
threshold is visible; the same adjustment was made for the 3-wheel-vs-
4-wheel utilization figure (200 orbits). (2) The gravity-gradient
component's second-harmonic oscillation (§1) was confirmed as real
physics (via the $\hat r_b\times I\hat r_b$ quadratic-form argument)
before being described as such, rather than assumed to be a plotting
artifact.

## 10. Limitations

- All disturbance models are simplified, illustrative approximations
  (§1) — not validated against a real spacecraft or mission environment.
- The orbital "propagation" here is timing-only (a fixed circular period
  and simplified periodic direction sweeps for gravity-gradient and
  magnetic field) — there is no real orbit/attitude propagator, no
  eclipse modeling, and no true IGRF magnetic field.
- SRP and aerodynamic directions are held fixed in the body frame
  (worst-case secular-bias simplification, §1), not derived from a
  rotating sun/velocity-vector geometry.
- The wheel-space allocation sign convention (§3) is a bookkeeping
  choice; magnitudes and comparative conclusions are convention-
  invariant, but the specific sign of any single wheel's momentum drift
  should not be over-interpreted physically.
- Momentum dumping/desaturation is **not implemented**. Every
  "saturation time" or "orbits to threshold" number in this milestone is
  the time until unloading becomes necessary, not a correction or
  recovery time — that is Milestone 4 scope.
- The mean-torque long-horizon estimate (§6) was validated against direct
  numerical integration for the nominal 4-wheel baseline only; a
  disturbance environment with a much larger periodic-to-secular ratio,
  or a periodic amplitude approaching $H_{\rm threshold}$, would need the
  same validation repeated (the machinery to do so, `first_threshold_
  crossing` on a full numerical integration, already exists and is used
  here as the check).

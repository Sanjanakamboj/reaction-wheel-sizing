# Milestone 4 — Momentum Dumping, Desaturation Logic & Schedule Methodology

This document explains the magnetorquer physics, the desaturation control
law (including its sign derivation), the hysteresis state machine, and
the hybrid long-duration scheduling approach implemented in
[`desaturation.py`](../src/reaction_wheel/desaturation.py). It builds on,
and does not duplicate, `geometry.py` (allocation, null space,
utilization), `disturbances.py` (magnetic-dipole torque, field model),
and `momentum.py` (mean-torque secular estimate). It does **not** cover
commercial hardware selection — that is Milestone 5+ scope.

## 1. Sign convention — derived, not guessed

M3 established that wheel-space allocation of *any* external body-frame
torque reuses `geometry.allocate_torque` unchanged:

$$\boldsymbol\tau_w = -A^{+}\boldsymbol\tau_{\rm external}, \qquad
\dot{\mathbf h}_w = \boldsymbol\tau_w.$$

For a magnetorquer unloading law to actually *decrease* stored wheel
momentum, the desired external unloading torque must be

$$\boxed{\boldsymbol\tau_{\rm unload,desired} = +k_H\,\boldsymbol H_w^{\rm body}}, \qquad
\boldsymbol H_w^{\rm body} = A\mathbf h_w,\ \ k_H > 0\ [\mathrm{s}^{-1}]$$

This is the **opposite sign** from a naive reading of "point the external
torque opposite the stored momentum" ($\propto -H_w^{\rm body}$). That
naive sign, substituted into the already-established
$\boldsymbol\tau_w=-A^+\boldsymbol\tau_{\rm external}$ relation, produces
$\dot{\mathbf h}_w = +k_H(A^+A)\mathbf h_w$ — **positive feedback**, growing
wheel momentum without bound. The $+k_H H_w^{\rm body}$ sign was verified
numerically *before* being adopted: for the orthogonal 3-wheel geometry
($A=I$, so $A^+A=I$ exactly) it gives $\dot{\mathbf h}_w=-k_H\mathbf h_w$,
clean exponential decay, exactly as intended. This derivation and its
numerical check are documented directly in the `desaturation.py` module
docstring and are also a named regression test
(`test_unload_sign_produces_decay_orthogonal`).

**A second consequence, load-bearing for §9 below**: for the redundant
4-wheel geometry, $\dot{\mathbf h}_w = -k_H(A^+A)\mathbf h_w$ depends
*only* on the row-space (body-momentum-observable) component of
$\mathbf h_w$ — it is *exactly zero* for any null-space component,
verified numerically
(`test_unload_only_affects_row_space_component_tetrahedral`). External
magnetorquer unloading can only ever remove $A\mathbf h_w$; it
structurally cannot touch a null-space wheel-momentum imbalance.

## 2. Magnetorquer physics and its fundamental limitation

A magnetic dipole $\mathbf m$ in field $\mathbf B$ produces torque
$\boldsymbol\tau_m = \mathbf m\times\mathbf B$. Since
$(\mathbf m\times\mathbf B)\cdot\mathbf B \equiv 0$ for any $\mathbf m$
(verified directly, `test_magnetic_torque_perpendicular_to_field`), **no
magnetic torque can ever be generated parallel to the instantaneous
field**. A magnetorquer is fundamentally a 2-axis (instantaneously),
not a 3-axis, actuator — full 3-axis authority over an orbit requires the
field direction to rotate relative to the body, which it does for a
non-equatorial, non-polar orbit (§4).

For a dipole capability $|\mathbf m|\le m_{\max}$, the maximum achievable
torque magnitude is $\tau_{m,\max}=m_{\max}|B|$, reached when
$\mathbf m\perp\mathbf B$ (`Magnetorquer.max_torque`, verified in
`test_maximum_torque_when_m_perpendicular_to_B`).

### Dipole inversion

Given a desired torque already projected onto the achievable plane
($\boldsymbol\tau_\perp = (I-\hat B\hat B^T)\boldsymbol\tau_{\rm desired}$,
`project_perpendicular_to_field`), the minimum-norm dipole command is

$$\mathbf m = \frac{\mathbf B\times\boldsymbol\tau_\perp}{|\mathbf B|^2}$$

(`dipole_command_for_torque`). This satisfies
$\mathbf m\times\mathbf B=\boldsymbol\tau_\perp$ exactly (vector
triple-product identity, verified to $<10^{-9}$ for 20 random seeded
directions). `Magnetorquer.saturate` then scales the commanded dipole
down (preserving direction) if it exceeds $m_{\max}$, or optionally
clips per-axis if a `m_max_per_axis` limit is set.

## 3. Representative magnetorquer

$m_{\max}=20$ A·m² — a representative small/medium smallsat-class
magnetorquer rod-set capability, **not a commercial product**. Against
the M3 baseline field ($|B|\approx3\times10^{-5}$ T), this gives a
maximum instantaneous torque authority of
$\tau_{m,\max}=6.0\times10^{-4}$ N·m — roughly **300× the mean total
environmental disturbance torque** ($\sim2\times10^{-6}$ N·m, M3), which
is why the resulting dump durations (§7) are short (hours) compared to
the multi-hundred-orbit accumulation timescale (M3).

## 4. Time-varying field model and orbit-averaged effectiveness

M4 reuses M3's `dipole_field_body` unchanged — a representative periodic
body-frame field model, **not a flight IGRF magnetic-field model**: a
constant-magnitude vector whose *direction* sweeps once per orbit with an
inclination-dependent tilt. Because the direction rotates, the
instantaneously-inaccessible component (parallel to $B$ at any one
instant, §2) becomes accessible at other points in the orbit.

The field-geometry effectiveness (independent of dipole saturation)

$$\eta_B = \frac{\lVert\boldsymbol\tau_\perp\rVert}{\lVert\boldsymbol\tau_{\rm desired}\rVert}
\in[0,1]$$

is exactly 0 when $\boldsymbol\tau_{\rm desired}\parallel\mathbf B$ and
exactly 1 when $\boldsymbol\tau_{\rm desired}\perp\mathbf B$ (both
verified directly). Over the baseline dump, $\eta_B$ ranges from 0.56 to
0.93 (mean 0.76) — the field geometry alone costs up to ~44% of the
desired torque at the worst instant, recovered as the field direction
rotates.

## 5. Operational thresholds and hysteresis

Separate dump-on / dump-off thresholds are used (never a single
chatter-prone threshold):

$$H_{\rm on}=0.8\,H_{\max}, \qquad H_{\rm off}=0.4\,H_{\max}$$

— representative operational choices (not derived or sourced), giving a
40%-of-$H_{\max}$ hysteresis band. $H_{\rm on}$ matches M3's operational
threshold exactly for continuity. `HysteresisThresholds` enforces
$0<H_{\rm off}<H_{\rm on}$ at construction. The state machine
(`DesatState`, `next_state`) has exactly two states —
**ACCUMULATING** and **DESATURATING** — with transitions gated on
worst-wheel utilization $\max_i|h_i|$ (a **per-wheel** trigger, per the
M4 spec: real wheel hardware saturates individually, so body-space net
momentum being small is not by itself a safe release condition — verified
directly: `test_dump_terminates_only_when_every_wheel_below_off`). Any
value strictly between $H_{\rm off}$ and $H_{\rm on}$ never triggers a
transition regardless of the current state — this is precisely what
prevents chatter (`test_no_chatter_in_hysteresis_band`).

## 6. Closed-loop dump simulation

At each time step, `simulate_dump` computes the unload command from the
*current* wheel momentum (state feedback), combines it with the active
M3 disturbance torque, allocates the total external torque via
`geometry.allocate_torque` (reused, not duplicated), and integrates
trapezoidally — mirroring M3's numerical-integration style. The dump
terminates only when **every** wheel satisfies $|h_i|\le H_{\rm off}$
(§5), never merely on body-space momentum. If the horizon `t_max` elapses
first, `reached_off=False` is returned — never a fabricated duration.

Two controlled analytical verification cases are included, matching the
M4 spec precisely:

- **Constant field, perpendicular momentum**: with a *fixed* (non-
  proportional) achievable torque, momentum decreases *linearly*,
  $H(t)=H_0-\tau_{\rm unload}t$ — verified against
  `momentum.integrate_wheel_momentum` to $<10^{-6}$ N·m·s
  (`test_constant_field_perpendicular_momentum_linear_decay`).
- **Momentum parallel to $B$**: the instantaneous achievable torque is
  exactly zero (`test_no_authority_parallel_to_field`) — demonstrating
  directly why the field's orbital rotation (§4) is operationally
  necessary.

## 7. Baseline single-dump result

Starting from the natural secular-accumulation direction scaled so the
worst wheel sits at $H_{\rm on}$ (nominal 4-wheel tetrahedral):

| Quantity | Value |
|---|---|
| Initial wheel momentum | $[7.20,\,-10.05,\,-4.60,\,7.46]$ N·m·s |
| Dump duration | 20,380 s = 3.59 orbits = 5.66 hours |
| Ideal (unsaturated, $\eta_B{=}1$) estimate | 1386 s = 0.24 orbits |
| Final wheel momentum | $[3.32,\,-5.03,\,-1.65,\,3.35]$ N·m·s |
| Max / mean dipole command | 20.0 / 20.0 A·m² (saturated almost throughout) |
| Peak achieved unloading torque | $6.0\times10^{-4}$ N·m |
| Mean field effectiveness | 0.76 (range 0.56–0.93) |
| Max wheel-torque utilization during dump | 0.0022 (feasible) |

The actual dump takes **~15× longer** than the idealized unsaturated
exponential estimate. Two honestly-reported reasons, both readable
directly from the dump history: (1) mean field effectiveness is 0.76, not
1.0 (§4); (2) the commanded dipole is **saturated at $m_{\max}$ for
nearly the entire dump** (the proportional-gain law's unsaturated demand,
$k_H|H_w^{\rm body}|$, exceeds $m_{\max}|B|$ whenever $|H_w^{\rm body}|$ is
large — i.e. for most of this dump, so it behaves like a quasi-constant-
maximum-torque "bang-bang" unloading rather than a smooth exponential).
This is a genuine, reported finding, not an idealization.

## 8. Repeat interval, duty cycle, and mission schedule

For an approximately constant secular per-wheel torque, the analytical
repeat interval over the hysteresis band is

$$T_{\rm repeat} \approx \frac{H_{\rm on}-H_{\rm off}}{|\dot h_{\rm secular}|}$$

(`analytical_repeat_interval`) — for the nominal 4-wheel case, 779.3
orbits (51.2 days), giving a single-cycle duty cycle of **0.459%**. This
is the $(H_{\rm on}-H_{\rm off})/H_{\rm on}=50\%$ **band fraction** of
M3's full zero-to-threshold time (1558.5 orbits), exactly as expected
since accumulation is linear.

### Hybrid long-duration simulation

Simulating a full year (5555 orbits) at fine time resolution purely to
wait out ~780-orbit accumulation intervals would integrate millions of
unnecessary steps. `simulate_mission_schedule` instead:

- **ACCUMULATING**: advances *analytically* using the mean secular
  per-wheel torque (`momentum.mean_wheel_torque`) — exact for linear
  constant-torque accumulation, and previously validated against full
  numerical integration to **0.03%** at this scale (Milestone 3, carried
  forward unchanged rather than re-validated from scratch).
- **DESATURATING**: full closed-loop time-domain simulation
  (`simulate_dump`), since this is where the interesting fast dynamics
  (field rotation, saturation) actually happen.

Over a representative 1-year horizon: **6 dumps**, mean interval 51.45
days (min/max 51.44/51.46 — a stationary environment gives nearly
identical intervals, reported as such rather than manufacturing false
variation), total desaturation time 31.2 hours, **duty cycle 0.356%**.

### Reconciling "6 dumps/year" (M4) with "~7.1 dumps/year" (M5)

These are two different, individually correct quantities, not a bug:

- **M4's "6 dumps"** counts actual completed events in one concrete
  simulated year that **starts from zero wheel momentum**
  (`simulate_mission_schedule(..., h0=None)`). The *first* cycle in that
  year is a full zero-to-$H_{\rm on}$ charge-up — the same ~1550-orbit
  timescale as M3's zero-to-threshold result — which is substantially
  longer than a steady-state $H_{\rm off}\to H_{\rm on}$ cycle. That one
  long first cycle reduces how many complete cycles fit in 365 days.
- **M5's "~7.1 dumps/year"** is a **steady-state average rate**,
  `365 days / (repeat_interval + dump_duration)`, i.e. the long-run
  cadence *after* the initial charge-up, with no zero-momentum start-up
  transient included.

A mission that starts with wheels already near their operating band
(the realistic case after initial commissioning, not literally
momentum-free) would see the M5 steady-state rate from day one; a
mission counted from a literal zero-momentum epoch sees the M4 number in
its first year and the steady-state rate thereafter. Both scripts now
print this reconciliation explicitly rather than leaving the two numbers
to be compared without context.

## 9. Null-space redistribution vs. external unloading

M2 identified a 1-D null space for the 4-wheel tetrahedral geometry.
Adding any null-space vector to $\mathbf h_w$ leaves $A\mathbf h_w$
*exactly* unchanged (`null_space_redistribute`, verified to
$<10^{-9}$ N·m·s for several coefficients) — internal redistribution
*cannot*, by itself, remove any spacecraft angular momentum, because it
never touches the only quantity (body-frame observable momentum) that
matters for the total system's angular-momentum budget. Only an external
torque (magnetorquer here) reduces $A\mathbf h_w$, demonstrated directly:
applying one dump reduces $|A\mathbf h_w|$ from 17.50 to 8.18 N·m·s. This
distinction (§1, second consequence) is not incidental — it is the same
mathematical fact ($\dot{\mathbf h}_w$ from unloading depends only on the
row-space component of $\mathbf h_w$) viewed from two directions.

Null-space redistribution *could*, in principle, be used to delay an
*individual* wheel's threshold crossing by moving momentum toward wheels
with more headroom — this is analyzed as a mathematical possibility here
(the invariance proof) but **not implemented as an active control law**
in M4; doing so usefully would require deciding a redistribution
objective and schedule, which is future-milestone scope.

## 10. Configuration and sensitivity results

Full numbers are in
[`results/desaturation_table.md`](../results/desaturation_table.md) and
the M4 script's printed report; headline comparisons:

- **3-wheel vs. 4-wheel**: 4-wheel gives a longer repeat interval (779 vs.
  463 orbits) — consistent with M3's momentum-accumulation-lifetime
  comparison — at the cost of a longer individual dump (3.59 vs. 1.97
  orbits, since more total momentum accumulates in the wider 4-wheel
  band before triggering).
- **Single-wheel failure**: all four failure repeat intervals (454, 447,
  650, and 447 orbits) are shorter than the nominal 4-wheel value (779
  orbits) — a real degradation in every case — but they are **not
  symmetric** (W3's failure gives ~650 orbits, noticeably better than the
  other three at ~450), mirroring M3's finding that a fixed disturbance
  direction breaks the tetrahedral geometry's perfect capability symmetry
  (M2). Wheel-torque utilization during the dump remains comfortably
  feasible in every failure case (max 0.0037).
- **Magnetorquer capability sweep** (0.5×–4× baseline): dump duration
  scales close to inversely with $m_{\max}$ (6.88 → 3.59 → 1.79 → 0.89
  orbits) because the dump spends nearly all its duration saturated
  (§7) — an approximately constant-torque regime, so duration
  $\propto 1/\tau_{\rm achieved}\propto 1/m_{\max}$ is the expected
  scaling, confirmed rather than assumed.
- **Threshold-band trade**: repeat interval depends on the *band width*
  $H_{\rm on}-H_{\rm off}$, not its absolute position — three bands of
  equal width (0.9/0.5, 0.8/0.4, 0.7/0.3) give essentially the same
  repeat interval (~779 orbits), while a narrower band (0.8/0.6, half
  the width) gives almost exactly half the repeat interval (390 orbits).
  Duty cycle stays roughly constant (~0.46%) across all four cases in
  this regime, since dump duration scales down with the smaller
  momentum swing at a similar rate to the repeat interval.

## 11. Limitations

- $k_H$, $H_{\rm on}$, $H_{\rm off}$, and $m_{\max}$ are representative
  engineering choices, not derived from a specific mission's attitude-
  determination, power, or thermal constraints.
- The unloading law is a simple proportional feedback on body-frame wheel
  momentum; no integral/derivative terms, no rate limiting beyond
  dipole saturation, and no interaction with a real attitude controller
  (M4 assumes ideal attitude hold throughout, per the M1-M3 convention).
- The geomagnetic field model (§4) is a simplified periodic direction
  sweep, not a flight IGRF model — see `docs/momentum_accumulation_
  methodology.md` §1 for the same caveat as it applies to M3.
- The hybrid mission-schedule simulation's accumulation phase inherits
  M3's validated 0.03% mean-torque-approximation accuracy; it was not
  independently re-validated for M4's specific disturbance-plus-dump
  composition (the dump phase itself is always full closed-loop, so this
  affects only the ACCUMULATING segments).
- Null-space redistribution is analyzed (invariance proof) but not
  implemented as an active control law (§9).
- No commercial magnetorquer or reaction-wheel selection, no thermal or
  power-system interaction, and no fault-management logic beyond the
  single-wheel-failure cases already carried from M2/M3.

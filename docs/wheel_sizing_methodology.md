# Milestone 1 — Reaction-Wheel Mechanics & Maneuver Sizing Methodology

This document explains the physics, derivations, and verification approach
used in Milestone 1. It does **not** cover momentum dumping/desaturation —
that is Milestone 4 scope. See [`conventions.md`](conventions.md) for the
frozen frame/sign/unit conventions referenced throughout.

## 1. Reaction-wheel mechanics

A reaction wheel is a momentum-exchange device: a motor-driven rotor of
inertia $J_w$ spinning at speed $\Omega_w$ (relative to the spacecraft body,
about the wheel's spin axis) stores angular momentum

$$H_w = J_w \Omega_w.$$

Commanding a motor torque $\tau_w$ on the rotor changes the wheel speed
according to

$$\tau_w = J_w \dot{\Omega}_w \quad\Longrightarrow\quad \dot{\Omega}_w = \tau_w / J_w.$$

By Newton's third law, the wheel's stator (bolted to the spacecraft
structure) — and therefore the spacecraft body — feels the **equal and
opposite** reaction torque, $\tau_{\rm body} = -\tau_w$. This sign flip
(frozen in `conventions.md` §3.3) is the mechanism by which a reaction
wheel controls spacecraft attitude without expending propellant: the wheel
absorbs (or gives up) angular momentum, and the spacecraft body gives up
(or absorbs) the exact opposite amount, so total system angular momentum
is conserved for an isolated spacecraft+wheel system with no external
torque.

## 2. Torque sign convention

`wheel.py` operates entirely in **wheel-frame convention**: a positive
`tau_w` increases `Omega_w` and `H_w`, consistent with $H_w = J_w\Omega_w$.

`maneuvers.py` and `sizing.py` operate in **body-torque convention**: the
"required torque" for a maneuver is the torque needed *on the spacecraft*.
Converting a body-torque requirement into the corresponding wheel command
requires negating it: `wheel_torque_cmd = -tau_req`. This project keeps the
two conventions in physically separate modules specifically so this sign
flip cannot be silently lost — see `SlewRequirement.wheel_torque_cmd` and
`.delta_H_wheel` in `maneuvers.py`.

## 3. Rest-to-rest maneuver derivation (triangular-rate profile)

For a single-axis, rest-to-rest slew of commanded angle $\theta$ executed
in total time $T$, the simplest physically realizable open-loop profile is
bang-bang (triangular-rate): constant angular acceleration $\alpha$ for the
first half of the maneuver, and equal-magnitude deceleration for the second
half. Integrating $\dot\omega = \alpha$ twice from rest over $[0, T/2]$
gives the angle swept in the acceleration phase:

$$\theta/2 = \alpha (T/2)^2 / 2$$

which, doubled for both phases (both phases sweep an equal angle by
symmetry) gives

$$\theta = \alpha \left(\frac{T}{2}\right)^2 \quad\Longrightarrow\quad
\alpha = \frac{4\theta}{T^2}.$$

The required body torque follows directly from $\tau = I\alpha$:

$$\tau_{\rm req} = I\alpha = \frac{4 I \theta}{T^2}.$$

Peak body rate is reached at the midpoint $t=T/2$:

$$\omega_{\rm peak} = \alpha \cdot \frac{T}{2} = \frac{2\theta}{T}.$$

This is implemented in `maneuvers.triangular_slew_requirement`.

## 4. Angular-momentum exchange and its independent verification

At peak body rate, the spacecraft carries body angular momentum

$$H_{\rm body,peak} = I\,\omega_{\rm peak}.$$

For the isolated spacecraft+wheel system (no external torque), the wheel
must have absorbed the exact opposite momentum:

$$\Delta H_w = -H_{\rm body,peak}.$$

Milestone 1 verifies $|\Delta H_w| \approx I|\omega_{\rm peak}|$ two
independent ways:

1. **Body-momentum path** — evaluate $I \cdot \omega_{\rm peak}$ directly
   from the analytical triangular-rate solution.
2. **Torque-integration path** — integrate the (sign-flipped) wheel torque
   command over the acceleration half-phase, $\Delta H_w = \int_0^{T/2}
   \tau_w\,dt$, which for the constant-torque bang-bang profile reduces to
   $\tau_w \cdot T/2$.

Both paths are computed in `scripts/verify_wheel_sizing.py` and agree to
better than $3\times10^{-4}$ N·m·s (limited by the numerical integration
grid, not by the underlying physics — see §6).

A third, fully independent check integrates the nonlinear ODEs directly
(`maneuvers.propagate_rest_to_rest_slew`, `scipy.integrate.solve_ivp`) and
confirms the same peak momentum, plus that:

- the spacecraft returns to zero rate at $t=T$ (rest-to-rest);
- the spacecraft reaches the commanded angle $\theta$ at $t=T$;
- the wheel momentum returns to (approximately) zero at $t=T$ — consistent
  with total momentum conservation at an initial value of zero;
- total system angular momentum $H_{\rm total} = H_{\rm body} + H_w$ stays
  constant (to numerical-solver precision, ~$10^{-15}$ N·m·s) throughout
  the maneuver.

## 5. Torque sizing vs. momentum-storage sizing are different requirements

The scaling laws

$$\tau_{\rm req} \propto \theta/T^2 \qquad H_{\rm req} \propto \theta/T$$

mean torque and momentum requirements do **not** scale the same way with
maneuver time. Concretely:

- **Doubling the maneuver angle** at fixed duration doubles *both*
  $\tau_{\rm req}$ and $H_{\rm req}$.
- **Doubling the maneuver duration** at fixed angle divides $\tau_{\rm
  req}$ by 4 but only divides $H_{\rm req}$ by 2.

This means a wheel set sized purely for momentum storage (e.g., to survive
a slow, large-angle slew) is not automatically adequate for a fast,
small-angle slew, and vice versa — torque capability and momentum-storage
capacity are two independent actuator specifications that must each be
checked (`sizing.size_single_axis_maneuver` reports both `rho_tau` and
`rho_H`, and the `active_constraint` field names whichever is more
restrictive for a given candidate wheel).

## 6. Wheel-speed and wheel-acceleration requirements

Given a required momentum excursion $H_{\rm req}$ and a reference rotor
inertia $J_w$, the required wheel-speed excursion is

$$\Omega_{\rm req} = H_{\rm req}/J_w,$$

and given a required torque $\tau_{\rm req}$, the required wheel
acceleration is

$$\dot\Omega_{w,\rm req} = \tau_{\rm req}/J_w.$$

Because the triangular-rate profile holds $\dot\Omega_{w,\rm req}$
constant over the acceleration half-phase $[0,T/2]$, integrating it over
that phase reproduces the required speed excursion exactly:

$$\Omega_{\rm req} = \dot\Omega_{w,\rm req}\cdot\frac{T}{2}.$$

This identity is checked directly in `tests/test_sizing.py` and gives a
second, algebraically independent path (distinct from
$\Omega_{\rm req}=H_{\rm req}/J_w$) to the same wheel-speed requirement.

## 7. Sizing margins

Torque capability and momentum-storage capacity are given **separate**
illustrative margin factors, $SF_\tau$ and $SF_H$ (default 1.5 each in
this repository — see `sizing.SizingMargins`), rather than one blanket
"safety factor." These are representative engineering placeholders, not
adopted mission requirements; a real program would set them per its own
verification, thermal derating, wheel-aging, and control-authority
margin policy.

## 8. Candidate-capability checks

For a candidate wheel with capability $(\tau_{\max}, H_{\max},
\Omega_{\max})$, this project computes capability *ratios*, not margins:

$$\rho_\tau = \frac{\tau_{\rm req}}{\tau_{\max}}, \qquad
\rho_H = \frac{H_{\rm req}}{H_{\max}}, \qquad
\rho_\Omega = \frac{\Omega_{\rm req}}{\Omega_{\max}}.$$

$\rho < 1$ means the candidate satisfies that particular requirement;
$\rho = 1$ is exactly at the limit; $\rho > 1$ means the candidate is
insufficient. These ratios are conceptually distinct from the *design
margins* $SF_\tau, SF_H$ of §7 — margins scale up the *requirement* before
a wheel is chosen; capability ratios compare a requirement against an
*already-specified* wheel's rated capability. The two should not be
conflated.

## 9. What Milestone 1 deliberately does not do

- It does not select a commercial/flight reaction wheel. The
  "representative wheel" in `wheel.representative_wheel()` is a synthetic
  capability model used only to exercise the mechanics and the
  candidate-capability-ratio logic end to end.
- It does not model 3-axis wheel geometry, wheel-to-body torque
  allocation, or worst-wheel loading (Milestone 2).
- It does not model environmental disturbance torques or momentum
  accumulation over an orbit (Milestone 3).
- It does not model momentum dumping/desaturation (Milestone 4).

## 10. Verification approach summary

| Check | Method | Result (representative case, 60°/60 s about Iy) |
|---|---|---|
| Attitude angle reaches command | Numerical ODE vs. analytical $\theta$ | residual $\approx 1.4\times10^{-10}$ rad |
| Final rate returns to zero | Numerical ODE | residual $\approx 4.6\times10^{-12}$ rad/s |
| Peak rate matches analytical | Numerical ODE vs. $\omega_{\rm peak}=2\theta/T$ | residual $\approx 8.7\times10^{-6}$ rad/s |
| Peak wheel momentum matches analytical | Numerical ODE vs. $I\omega_{\rm peak}$ | residual $\approx 2.4\times10^{-4}$ N·m·s |
| Torque-integration vs. momentum-requirement | $\int\tau_w\,dt$ vs. $I\omega_{\rm peak}$ | residual $\approx 2.4\times10^{-4}$ N·m·s |
| Total angular momentum conserved | $H_{\rm body}+H_w$ over full maneuver | max $\lvert H_{\rm total}\rvert \approx 4.6\times10^{-15}$ N·m·s |
| Angle-doubling scaling law | $\tau,H$ ratio at fixed $T$ | exactly 2.0000, 2.0000 |
| Time-doubling scaling law | $\tau,H$ ratio at fixed $\theta$ | exactly 0.2500, 0.5000 |

Exact numbers are regenerated by `scripts/verify_wheel_sizing.py` and are
also captured as automated regression tests in `tests/test_maneuvers.py`
and `tests/test_sizing.py`.

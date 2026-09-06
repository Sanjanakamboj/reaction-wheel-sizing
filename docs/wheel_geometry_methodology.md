# Milestone 2 — Wheel-Set Geometry, Allocation & Redundancy Methodology

This document explains the geometry, allocation math, capability envelope,
and single-wheel-failure analysis implemented in
[`src/reaction_wheel/geometry.py`](../src/reaction_wheel/geometry.py). It
builds on, and does not duplicate, the single-axis wheel mechanics of
[`wheel.py`](../src/reaction_wheel/wheel.py) and the frozen conventions in
[`conventions.md`](conventions.md). It does **not** cover environmental
disturbance torques or desaturation — those are Milestone 3/4 scope.

## 1. Wheel-axis matrix

For $N$ reaction wheels, each wheel $i$ has a body-frame unit spin-axis
vector $\hat a_i \in \mathbb{R}^3$. Collecting them as columns gives the
wheel-axis matrix

$$
A = [\hat a_1 \mid \hat a_2 \mid \cdots \mid \hat a_N] \in \mathbb{R}^{3\times N}.
$$

`WheelSetGeometry` validates, at construction, that: the input has shape
$(3,N)$; every value is finite; every column is unit-norm to $10^{-8}$;
and $N \ge 3$ (fewer than 3 wheels cannot span 3-axis body torque/momentum
at all). **Badly-scaled axes are rejected, never silently renormalized** —
if you want a scaled input treated as a direction, normalize it yourself
before constructing the geometry.

## 2. Sign convention — derived, not assumed

M1 established the scalar relation $\tau_{\rm body} = -\tau_w$ for a
single wheel aligned with a body axis (`docs/conventions.md` §3.3). For a
wheel $i$ with an arbitrary spin-axis direction $\hat a_i$, the same
Newton's-third-law argument applies **along that wheel's own physical spin
axis**: the motor applies $\tau_{w,i}$ to the rotor, and the stator (and
therefore the spacecraft) feels the reaction $-\tau_{w,i}$, directed along
the same axis $\hat a_i$. Summing every wheel's contribution to the net
body torque:

$$
\boldsymbol{\tau}_{\rm body} = \sum_{i=1}^N \left(-\tau_{w,i}\right)\hat a_i
= -A\,\boldsymbol{\tau}_w.
$$

Setting $N=1$ (or, since this project requires $N\ge3$, restricting to a
single active wheel in an orthogonal 3-wheel set with $A=I_3$) recovers
M1's scalar relation exactly — this is checked directly as a regression
test (`test_geometry.py::test_single_wheel_matches_m1_torque_and_momentum_sign`,
`test_orthogonal_reproduces_m1_maneuver_requirement_exactly`).

The identical structure holds for angular momentum. Each wheel stores
scalar momentum $h_{w,i} = J_{w,i}\Omega_i$ (from `wheel.py`, applied
per-wheel), and its body-frame contribution is $h_{w,i}\hat a_i$, so the
wheel-set's total body-frame angular momentum is

$$
H_w^{\rm vec} = A\,\mathbf{h}_w, \qquad
\boldsymbol{H}_{\rm body} + A\,\mathbf{h}_w = \text{constant}
$$

for an isolated spacecraft + wheel-set (no external torque) — the direct
vector generalization of M1's $H_{\rm body} + H_w = \text{const}$.

## 3. Standard geometries

### 3-wheel orthogonal (baseline)

$$A_3 = I_3$$

Wheel $x$/$y$/$z$ aligned one-to-one with body $x$/$y$/$z$. A pure
principal-axis torque/momentum command activates exactly one wheel and
exactly reproduces the corresponding M1 scalar result.

### 4-wheel tetrahedral (redundant)

$$
A_4 = \frac{1}{\sqrt3}
\begin{bmatrix}
 1 &  1 & -1 & -1\\
 1 & -1 &  1 & -1\\
 1 & -1 & -1 &  1
\end{bmatrix}
$$

(columns proportional to $[1,1,1]$, $[1,-1,-1]$, $[-1,1,-1]$,
$[-1,-1,1]$, each normalized). This is a maximally symmetric arrangement:
all four singular values of $\sigma(A_4)$ are equal ($\sigma =
2/\sqrt3\approx1.155$ for each of the 3 nonzero singular values), so
$\kappa(A_4)=1$ exactly — same as the orthogonal geometry — and $A_4$ is
a **tight frame**: $A_4 A_4^T = \frac{4}{3}I_3$. This is why, as shown
below, the 4-wheel geometry's directional torque capability is *exactly*
$4/3$ that of the 3-wheel orthogonal geometry in *every* direction, not
just some.

Removing any one column leaves the other three spanning $\mathbb{R}^3$
(rank 3, verified for all four choices in
`test_tetrahedral_single_failure_preserves_full_rank`), so the geometry
provides one-wheel fault tolerance.

## 4. Torque and momentum allocation

For a commanded body torque $\boldsymbol\tau_c$, the minimum-Euclidean-norm
wheel-torque solution to $\boldsymbol\tau_c = -A\boldsymbol\tau_w$ uses the
Moore-Penrose pseudoinverse:

$$
\boldsymbol\tau_w = -A^{+}\boldsymbol\tau_c.
$$

For a full-row-rank $A$ this reconstructs $\boldsymbol\tau_c$ to
floating-point precision (`allocate_torque`, verified for pure-axis,
equal-axis, mixed-sign, and 20 random seeded directions in
`test_geometry.py`). No generic optimizer is used — the unconstrained
pseudoinverse solution is closed-form and sufficient for this milestone.

Momentum allocation mirrors this exactly, solving for the wheel-momentum
vector required to realize a commanded **body momentum change**
$\Delta\boldsymbol H_{\rm body}$ consistent with conservation
($A\mathbf h_w = -\Delta\boldsymbol H_{\rm body}$):

$$
\mathbf h_w = -A^{+}\Delta\boldsymbol H_{\rm body}.
$$

Both `allocate_torque` and `allocate_momentum` use the identical sign
pattern, so the orthogonal geometry reproduces M1's numbers exactly on
both the torque and the momentum side (see the M1 regression check in
`scripts/analyze_wheel_geometry.py`).

## 5. Null space (4-wheel only)

Since $A_4$ has 4 columns but rank 3, $\dim\mathcal N(A_4)=1$. Any vector
$\mathbf z\in\mathcal N(A_4)$ satisfies $A_4\mathbf z\approx 0$, so

$$
\boldsymbol\tau_w = -A_4^{+}\boldsymbol\tau_c + N z
$$

produces the *same* body torque $\boldsymbol\tau_c$ for any scalar $z$
(verified in `test_null_space_alternative_produces_same_body_torque` and
demonstrated numerically in the M2 script). This spare degree of freedom
is **not used for anything in this milestone** — no wheel-speed
balancing, no momentum redistribution, no desaturation — it is only shown
to exist as groundwork for later milestones that will use it for exactly
those purposes.

## 6. Per-wheel utilization and infeasibility

Given an allocation, per-wheel utilization ratios are

$$
\rho_{\tau,i} = \frac{|\tau_{w,i}|}{\tau_{{\max},i}}, \qquad
\rho_{H,i} = \frac{|h_{w,i}|}{H_{{\max},i}},
$$

with the worst-wheel loading being $\max_i \rho_{\tau,i}$ /
$\max_i\rho_{H,i}$. **If any $\rho_{\tau,i} > 1$ the minimum-norm
allocation is reported as infeasible and is never silently clipped** —
clipping one wheel's torque changes the realized net body torque away
from what was actually commanded, silently corrupting the maneuver. See
`describe_feasibility`.

## 7. Directional torque-capability envelope

For a unit body-torque direction $\hat u$, allocation is linear, so
scaling the demand by $\lambda$ scales the min-norm wheel-torque solution
by the same $\lambda$. The largest $\lambda$ keeping every wheel within
$\tau_{\max}$ is therefore available in closed form:

$$
\boldsymbol\tau_w(\hat u) = -A^{+}\hat u, \qquad
\lambda_{\max}(\hat u) = \frac{\tau_{\max}}{\max_i|\tau_{w,i}(\hat u)|}.
$$

`directional_capability_envelope` evaluates this over a deterministic,
roughly-uniform Fibonacci-sphere sample of directions
(`fibonacci_sphere_directions`) with no randomness, so results are
reproducible run to run.

## 8. Isotropy

$$
\eta_\tau = \frac{\tau_{\min}}{\tau_{\max,{\rm capability}}}, \qquad
\tau_{\min}=\min_{\hat u}\lambda_{\max}(\hat u), \quad
\tau_{\max,{\rm capability}}=\max_{\hat u}\lambda_{\max}(\hat u).
$$

Numerically (4000-direction sample, representative wheel $\tau_{\max}=0.2$
N·m): the 3-wheel orthogonal geometry gives $\eta\approx0.587$ (capability
varies between 0.200 N·m on-axis and 0.341 N·m along a body diagonal,
where **all three wheels can contribute simultaneously and their
Euclidean-norm demand is spread across three limited actuators**) and the
4-wheel tetrahedral geometry gives $\eta\approx0.587$ as well — nearly
identical isotropy — but with **every value scaled up by a factor of
$4/3$**, a direct consequence of $A_4$ being a tight frame ($A_4A_4^T =
\frac43 I_3$, §3). So for this specific symmetric tetrahedral geometry,
adding the 4th wheel does not change the *shape* of the directional
capability envelope, only its *size* (uniformly larger by 4/3). This is
NOT a general property of arbitrary 4-wheel geometries — it is a
consequence of the specific symmetric choice made here, and is verified
numerically rather than assumed.

## 9. Single-wheel-failure analysis

For each wheel $j$ removed from $A_4$, the remaining $3\times3$ matrix is
checked for rank (all four cases: rank 3, full 3-axis authority retained)
and its own directional capability envelope and condition number are
computed. By the tetrahedral geometry's symmetry, all four failure cases
are geometrically equivalent (verified: the spread across the four
$\tau_{\min}$ values after failure is $\sim10^{-5}$ N·m, i.e. numerical
noise, not a real difference).

## 10. Redundancy vs. nominal capability — a critical distinction

**Redundancy (surviving a wheel failure) and increased nominal capability
are not the same concept**, and this milestone deliberately quantifies
both rather than conflating them:

- The 4-wheel tetrahedral geometry's **nominal** minimum directional
  torque capability (0.267 N·m) *is* larger than the 3-wheel orthogonal
  geometry's (0.200 N·m) — a genuine capability gain from the 4th wheel,
  consistent with the tight-frame scaling of §8.
- But after **any single wheel fails**, the tetrahedral geometry's
  remaining capability drops to a mean $\tau_{\min}\approx0.163$ N·m — a
  **~38.8% loss** relative to its own nominal 4-wheel capability, and
  *below* the 3-wheel orthogonal baseline's nominal 0.200 N·m.
- In other words: the 4th wheel buys extra performance when all four
  wheels work, and buys fault tolerance (rank/authority preservation) when
  one fails — but it does **not** mean the failed-down 4-wheel system
  still outperforms a plain 3-wheel system. A program choosing 4 wheels
  for redundancy should size against the **failed-case** capability, not
  the nominal 4-wheel capability, if one-wheel-failure operation is a
  requirement.

## 11. Condition number — a caveat

The condition number $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ measures
how sensitive the minimum-norm allocation is to the *direction* of the
demand (a well-conditioned, isotropic geometry has $\kappa$ near 1); it
does **not** by itself say anything about wheel torque/momentum capability
in physical units. Both the orthogonal and nominal tetrahedral geometries
here have $\kappa=1$ (both are tight frames / orthogonal-columns
matrices), yet they have different absolute torque capability (§8) — the
condition number and the capability envelope are complementary, not
interchangeable, metrics.

## 12. Scope boundary

This milestone does not implement environmental disturbance torques,
orbit propagation, momentum accumulation, or desaturation — see the
project README for the full M2 scope boundary. The null-space freedom
(§5) is demonstrated but not yet used for any secondary objective.

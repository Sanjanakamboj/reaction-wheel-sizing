# GNC-04 — Reaction Wheel Sizing & Momentum Management

**Milestone 1: Reaction-Wheel Mechanics & Maneuver Torque Sizing**
**Milestone 2: 3-Axis Wheel-Set Geometry, Torque Allocation & Wheel Loading**

## Project objective

Build a rigorous, reproducible reaction-wheel sizing and
momentum-management analysis for a representative small spacecraft. The
final portfolio deliverable (across all milestones) is a sized
reaction-wheel set, a momentum-storage assessment, and a desaturation
schedule/strategy.

## Engineering problem

> **What reaction-wheel torque and momentum-storage capability are
> required to execute representative spacecraft attitude maneuvers while
> absorbing environmental disturbance momentum, and how frequently must
> the wheel set be desaturated?**

This is a **reaction-wheel sizing and momentum-management** project, not
an attitude-controller-design project. Controller design appears only
insofar as a physically defined maneuver torque profile (an open-loop,
bang-bang rest-to-rest slew) is needed to drive actuator sizing.

Milestone 1 establishes the foundational wheel mechanics and
single-axis maneuver-driven torque/momentum sizing, with independent
analytical and numerical verification. It does **not** yet cover 3-axis
wheel geometry, environmental disturbance torques, or momentum dumping —
those are later milestones (see [Planned next milestone](#planned-next-milestone)).

## Representative spacecraft

Illustrative engineering assumptions, not a specific spacecraft — a
~180 kg smallsat-class bus with an asymmetric diagonal inertia tensor:

| Axis | Inertia [kg·m²] |
|---|---|
| $I_x$ | 18.0 |
| $I_y$ | 28.0 |
| $I_z$ | 22.0 |

Validated at construction for finiteness, strict positivity, and
(diagonal) positive-definiteness — see [`spacecraft.py`](src/reaction_wheel/spacecraft.py).

## Governing equations

Wheel mechanics (wheel-frame convention):

$$H_w = J_w\Omega_w \qquad H_{\max} = J_w\Omega_{\max} \qquad \dot\Omega_w = \tau_w/J_w$$

Rest-to-rest triangular-rate slew of angle $\theta$ in time $T$
(body-torque convention):

$$\alpha = \frac{4\theta}{T^2} \qquad \tau_{\rm req} = I\alpha \qquad \omega_{\rm peak} = \frac{2\theta}{T} \qquad H_{\rm req} = I\,\omega_{\rm peak}$$

Scaling laws: $\tau_{\rm req}\propto\theta/T^2$, $H_{\rm req}\propto\theta/T$
— torque and momentum sizing are **different actuator requirements** that
do not scale the same way with maneuver time. Sign conventions (wheel
torque vs. body reaction torque, momentum conservation) are frozen in
[`docs/conventions.md`](docs/conventions.md); full derivations are in
[`docs/wheel_sizing_methodology.md`](docs/wheel_sizing_methodology.md).

## M1 verification

`scripts/verify_wheel_sizing.py` runs the full M1 report: spacecraft/wheel
summary, a 15-row maneuver-sizing table (5 cases × 3 axes), analytical-vs-
numerical residuals, total-angular-momentum conservation, and scaling-law
checks — then generates three figures.

| Check | Result |
|---|---|
| Analytical vs. numerical final angle | residual ≈ 1.4×10⁻¹⁰ rad |
| Analytical vs. numerical final rate | residual ≈ 4.6×10⁻¹² rad/s |
| Analytical vs. numerical peak momentum | residual ≈ 2.4×10⁻⁴ N·m·s |
| Torque-integration vs. momentum-requirement | residual ≈ 2.4×10⁻⁴ N·m·s |
| Total angular momentum conservation | max\|H_total\| ≈ 4.6×10⁻¹⁵ N·m·s |
| Angle-doubling scaling law (τ, H) | exactly 2.0000, 2.0000 |
| Time-doubling scaling law (τ, H) | exactly 0.2500, 0.5000 |

**Figures** (`results/`):

- `fig1_rest_to_rest_maneuver.png` — attitude angle, body rate, applied
  body torque for a 60°/60 s slew.
- `fig2_momentum_exchange.png` — spacecraft, wheel, and total angular
  momentum; total is flat at zero, visually confirming conservation.
- `fig3_torque_momentum_vs_duration.png` — required torque and momentum
  vs. maneuver duration, normalized on one axis to make the $T^{-2}$ vs.
  $T^{-1}$ slope difference directly visible, plus an absolute-units panel.

## Maneuver-sizing table

Full table: [`results/maneuver_sizing_table.md`](results/maneuver_sizing_table.md)
(5 representative maneuvers × 3 spacecraft axes, checked against a
synthetic representative wheel: $J_w=0.02$ kg·m², $\tau_{\max}=0.2$ N·m,
$\Omega_{\max}=6000$ rpm).

## Key findings

- **Peak-torque driver**: the aggressive 45°/10 s case about the $I_y$
  axis (0.880 N·m required) — and it *exceeds* the representative wheel's
  torque capability ($\rho_\tau = 4.40$), while comfortably satisfying
  momentum ($\rho_H = 0.35$). This is a clear demonstration that torque
  and momentum are independent constraints: the same maneuver can be
  momentum-feasible yet torque-infeasible.
- **Peak-momentum driver**: the same aggressive case, same axis (4.40
  N·m·s required).
- **Worst axis by inertia**: $I_y$ (28.0 kg·m²) drives both the largest
  torque and momentum requirement at fixed maneuver kinematics, as
  expected since $\tau,H \propto I$.
- All five representative maneuvers are **torque-limited** (not
  momentum- or speed-limited) against the representative wheel — i.e.
  $\rho_\tau > \rho_H, \rho_\Omega$ in every row of the sizing table.
- Doubling maneuver time cuts torque demand by 4× but momentum demand by
  only 2× — slowing down a maneuver is a much more effective lever for
  relaxing torque requirements than for relaxing momentum-storage
  requirements.

## Milestone 2 — 3-Axis Wheel-Set Geometry, Allocation & Redundancy

> **How does a multi-wheel geometry map spacecraft torque and angular
> momentum into individual wheel demands, and how do 3-wheel orthogonal
> and 4-wheel redundant configurations compare in worst-wheel loading and
> failure tolerance?**

M2 extends M1's single-axis mechanics to arbitrary wheel-axis geometries
via a wheel-axis matrix $A\in\mathbb R^{3\times N}$ (columns = body-frame
unit spin axes) and minimum-norm pseudoinverse allocation. Full
derivations, sign-convention reconciliation with M1, and the
redundancy-vs-capability distinction are in
[`docs/wheel_geometry_methodology.md`](docs/wheel_geometry_methodology.md).

**3-wheel orthogonal** ($A_3=I_3$): baseline, one wheel per body axis.
**4-wheel tetrahedral** (redundant): axes proportional to
$[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]$, each normalized — a symmetric
tight frame ($A_4A_4^T=\tfrac43 I_3$) with $\kappa(A_4)=1$ and a 1-D null
space.

**Representative allocation** (M1's worst-case maneuver, 45°/10 s about
$I_y$, $\tau_{\rm req}=0.880$ N·m, mapped as a pure-$y$ body torque):

| Geometry | Wheel torques [N·m] | Worst wheel | $\rho_{\tau,\max}$ |
|---|---|---|---|
| 3-wheel orthogonal | [0, −0.880, 0] | Wy | 4.40 |
| 4-wheel tetrahedral | [−0.381, 0.381, −0.381, 0.381] | W1 | 1.90 |

Spreading the same body torque across 4 wheels roughly **halves** the
worst-wheel torque utilization relative to the 3-wheel case — but neither
configuration satisfies this particular (deliberately aggressive) demand
against the representative wheel's $\tau_{\max}=0.2$ N·m, correctly
flagged as infeasible rather than silently clipped. Full case-by-case
results (5 cases × 2 geometries) are in
[`results/wheel_loading_table.md`](results/wheel_loading_table.md).

**Torque-envelope comparison** — isotropy is nearly identical between the
two geometries ($\eta\approx0.587$ for both), but the tetrahedral
geometry's envelope is uniformly **4/3× larger** in every direction (a
direct consequence of its tight-frame structure):

| Geometry | $\tau_{\min}$ [N·m] | $\tau_{\max,{\rm cap}}$ [N·m] | $\eta=\tau_{\min}/\tau_{\max,{\rm cap}}$ |
|---|---|---|---|
| 3-wheel orthogonal | 0.200 | 0.341 | 0.587 |
| 4-wheel tetrahedral (nominal) | 0.267 | 0.455 | 0.587 |

**Single-wheel-failure result**: every one of the 4 possible single-wheel
failures leaves the remaining 3-wheel subset full rank (full 3-axis
control authority preserved), and by tetrahedral symmetry all four
failure cases are geometrically equivalent (spread in minimum capability
across the 4 cases: ~$10^{-5}$ N·m, i.e. numerical noise).

**Key redundancy finding**: nominal 4-wheel minimum capability (0.267
N·m) exceeds the 3-wheel baseline (0.200 N·m) — a real capability gain —
but after **any single wheel fails**, capability drops to a mean 0.163
N·m, a **38.8% loss relative to the nominal 4-wheel capability**, and
*below* the plain 3-wheel baseline. **Redundancy and increased nominal
capability are not the same concept**: a program requiring one-wheel-fault
tolerance must size against the failed-case capability, not the nominal
4-wheel number.

![Wheel-axis geometry](results/fig1_wheel_axis_geometry.png)
![Nominal vs single-wheel-failure capability](results/fig4_nominal_vs_failed_capability.png)

## Repository structure

```
reaction-wheel-sizing/
├── README.md
├── pyproject.toml
├── src/reaction_wheel/
│   ├── constants.py       # unit conversions (rpm <-> rad/s, deg <-> rad)
│   ├── spacecraft.py       # rigid spacecraft inertia model
│   ├── wheel.py            # ideal reaction-wheel mechanics model
│   ├── maneuvers.py        # rigid-body torque/momentum + triangular slew
│   ├── sizing.py           # reusable maneuver-driven sizing API
│   └── geometry.py         # multi-wheel geometry, allocation, capability (M2)
├── tests/                  # pytest suite (112 tests)
├── scripts/
│   ├── verify_wheel_sizing.py     # M1 verification report + figures + table
│   └── analyze_wheel_geometry.py  # M2 analysis report + figures + table
├── docs/
│   ├── conventions.md                 # frozen sign/unit/frame conventions
│   ├── wheel_sizing_methodology.md    # M1 derivations + verification approach
│   └── wheel_geometry_methodology.md  # M2 geometry/allocation/redundancy methodology
└── results/                # generated figures + sizing/loading tables
```

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/verify_wheel_sizing.py
python scripts/analyze_wheel_geometry.py
```

## Limitations

- No environmental disturbance torques (gravity-gradient, aerodynamic,
  solar-radiation-pressure, magnetic) or momentum accumulation over an
  orbit (Milestone 3).
- No momentum-dumping/desaturation modeling (Milestone 4).
- No commercial reaction-wheel selection — the "representative wheel" is
  a synthetic capability model used only to exercise the mechanics, and
  all wheels in the M2 sets share identical capability.
- Rest-to-rest maneuver kinematics use an idealized bang-bang
  (triangular-rate) open-loop profile, not a closed-loop controller; the
  3-axis combined slew case (M2 §15) uses a decoupled per-axis sizing
  approximation, not exact nonlinear rigid-body attitude dynamics.
- Spacecraft inertia is diagonal (principal-axis); no products of inertia.
- The M2 null-space freedom in the 4-wheel geometry is demonstrated but
  not yet used for any secondary objective (wheel-speed balancing,
  momentum redistribution) — that is future-milestone scope.
- M2 allocation is unconstrained minimum-norm; infeasible demands are
  detected and reported, never silently clipped.

## Planned next milestone

**M3 — Disturbance momentum accumulation**: representative environmental
disturbance torques (gravity-gradient, aerodynamic, solar-radiation-pressure,
magnetic), momentum buildup over an orbit, storage sizing, and saturation
time.

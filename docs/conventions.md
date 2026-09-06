# Engineering Conventions — GNC-04 Reaction Wheel Sizing

This document freezes the frames, sign conventions, and units used throughout
the project. Every module and script in this repository must be consistent
with these definitions. Where a formula could be ambiguous about sign, this
document is the tie-breaker.

## 1. Units

All internal calculations use SI units unless explicitly converted for
display:

| Quantity            | Unit      |
|---------------------|-----------|
| Length              | m         |
| Mass                | kg        |
| Time                | s         |
| Angle               | rad       |
| Angular rate         | rad/s     |
| Angular acceleration | rad/s²    |
| Torque              | N·m       |
| Angular momentum    | N·m·s     |
| Moment of inertia   | kg·m²     |

Angles and rates are converted to degrees / rpm **only** at the
display/reporting layer (tables, printed reports, plot labels). All
`reaction_wheel` library functions accept and return SI units.

Angular momentum unit identity used throughout:

$$
1\ \mathrm{N\,m\,s} = 1\ \mathrm{kg\,m^2/s}
$$

This follows directly from $H = I\omega$: $[\mathrm{kg\,m^2}][\mathrm{rad/s}]$
has the same dimensional content as $[\mathrm{N\,m}][\mathrm{s}]$ since
$\mathrm{N} = \mathrm{kg\,m/s^2}$ and radians are dimensionless.

## 2. Frames

- **Inertial frame** $\mathcal{N}$: a fixed (non-rotating) reference frame
  in which absolute attitude and absolute angular momentum are expressed
  for conservation checks. For the single-axis analyses in Milestone 1 the
  inertial frame and the body frame share a common axis of rotation, so the
  scalar projection of momentum/rate along that axis is frame-independent;
  the distinction matters starting in later milestones with multi-axis
  wheel geometry.
- **Spacecraft body frame** $\mathcal{B} = (\hat{b}_1, \hat{b}_2, \hat{b}_3)$
  ($x, y, z$): right-handed, principal-axis frame in which the spacecraft
  inertia tensor is diagonal for this milestone:

$$
I = \mathrm{diag}(I_x, I_y, I_z)
$$

- **Wheel spin axis**: for a single reaction wheel considered in isolation
  (Milestone 1), the wheel spin axis is aligned with one body axis
  (e.g. $\hat{b}_1$ for an "x-wheel"). The wheel's positive spin direction
  is defined as positive rotation about that body axis using the
  right-hand rule. Multi-wheel, non-body-aligned geometries (skewed/pyramid
  configurations) are introduced in Milestone 2.

## 3. Sign conventions

### 3.1 Spacecraft (body) angular momentum and rate

Positive spacecraft body rate $\omega$ and positive spacecraft angular
momentum $H_{\rm body} = I\omega$ about a given body axis follow the
right-hand rule about that axis, consistent with the inertial frame
defined in §2.

### 3.2 Wheel angular momentum and speed

For an ideal (symmetric, balanced) reaction wheel with rotor inertia
$J_w$ about its spin axis, spinning at wheel speed $\Omega_w$ (relative to
the spacecraft body, expressed along the same body axis as its spin axis):

$$
H_w = J_w \Omega_w
$$

Positive $\Omega_w$ and positive $H_w$ use the **same** right-hand-rule
sign as the body axis they are aligned with. A wheel spinning
"positively" about $+\hat{b}_1$ stores positive angular momentum along
$+\hat{b}_1$.

### 3.3 Wheel torque vs. reaction torque on the spacecraft

$\tau_w$ denotes the torque the motor applies **to the wheel rotor**:

$$
\tau_w = J_w \dot{\Omega}_w
$$

By Newton's third law, the wheel assembly (stator, bolted to the
spacecraft structure) feels the **equal and opposite** reaction torque.
The torque applied **to the spacecraft body** by the wheel is therefore:

$$
\tau_{\rm body} = -\tau_w = -J_w \dot{\Omega}_w
$$

**This sign flip is the single most important convention in this
project and must never be left implicit.** Consequences:

- To decelerate the spacecraft (apply negative body torque), the motor
  spins the wheel up in the positive direction ($\tau_w > 0 \Rightarrow
  \tau_{\rm body} < 0$).
- To accelerate the spacecraft in $+\hat{b}$, the motor commands
  $\tau_w < 0$ (spins the wheel toward negative $\Omega_w$, i.e. removes
  positive wheel momentum / adds negative wheel momentum).
- In this repository, `wheel.py` functions that compute $\dot\Omega_w$ or
  $H_w$ operate in **wheel-frame convention** ($\tau_w$, $\Omega_w$,
  $H_w$ all mutually consistent per $H_w = J_w\Omega_w$). Functions in
  `maneuvers.py` and `sizing.py` that describe a *commanded spacecraft
  maneuver* work in **body-torque convention** ($\tau_{\rm req}$ is the
  torque needed on the spacecraft) and explicitly negate when converting
  to the wheel-side torque/momentum command, i.e.:

$$
\tau_w^{\rm cmd} = -\tau_{\rm body}^{\rm req}, \qquad
\Delta H_w = -\Delta H_{\rm body}
$$

### 3.4 Isolated-system conservation

For the spacecraft + wheel treated as an isolated system (no external
environmental torque — the only case considered in Milestone 1), total
angular momentum along the shared axis is conserved:

$$
H_{\rm total} = H_{\rm body} + H_{w} = \text{constant}
$$

This is because the body torque and the wheel-reaction torque are
internal, equal-and-opposite action/reaction pairs and therefore cancel
in the total-system momentum budget. This identity is used directly as
a verification check in the M1 test suite and verification script.

## 4. Inertia convention

- The spacecraft inertia tensor $I$ is defined in the body frame about
  the spacecraft center of mass.
- Milestone 1 uses a diagonal (principal-axis) inertia tensor
  $I = \mathrm{diag}(I_x, I_y, I_z)$ with $I_x, I_y, I_z > 0$
  (validated at construction — see `spacecraft.py`).
- The spacecraft inertia used in single-axis maneuver sizing is the
  **spacecraft-only** inertia about the maneuver axis; wheel rotor
  inertia is small relative to spacecraft inertia and is not added to
  the spacecraft inertia in this milestone (consistent with a reaction
  wheel being a momentum-exchange device, not a mass added to the rigid
  body's slew inertia in this simplified single-axis treatment).

## 5. Torque command convention

A "required maneuver torque" $\tau_{\rm req}$, as computed in
`maneuvers.py`/`sizing.py`, is always expressed as the torque that must
be applied **to the spacecraft body** to achieve the commanded
kinematics ($\tau_{\rm req} = I\alpha$, §3.1 sign convention). The
corresponding **wheel torque command** is the negative of this
quantity (§3.3). All public sizing functions document which convention
(body-torque or wheel-torque) their return value uses.

## 6. Summary sign table

| Symbol | Meaning | Sign convention |
|---|---|---|
| $\omega$, $H_{\rm body}$ | Spacecraft body rate / momentum | RHR about body axis |
| $\Omega_w$, $H_w$ | Wheel speed / momentum | RHR about wheel spin axis (= body axis it's aligned with) |
| $\tau_w$ | Torque motor applies to wheel rotor | RHR about wheel spin axis |
| $\tau_{\rm body}$ | Reaction torque on spacecraft from wheel | $\tau_{\rm body} = -\tau_w$ |
| $\tau_{\rm req}$ | Required maneuver torque (body-frame) | RHR about maneuver axis |
| $H_{\rm total}$ | $H_{\rm body} + H_w$ | Conserved, isolated system |

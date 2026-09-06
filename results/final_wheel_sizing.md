# Milestone 5 — Final Reaction-Wheel Sizing Table

| Requirement source | Nominal requirement | Failed-wheel requirement | Selected margin | Final minimum capability | Active driver | Notes |
|---|---|---|---|---|---|---|
| Wheel torque | 0.0212 N*m | 0.0423 N*m | SF_tau=1.5 | **0.08 N*m** | One-wheel-failure-tolerant maneuver torque | Adopted 90deg/60s maneuver, worst axis, tetrahedral allocation |
| Momentum storage | n/a (headroom-driven) | 6.348 N*m*s (raw, no margin) | SF_H=1.5 | **12.566 N*m*s** | Maneuver-at-threshold headroom (failure case) | H_on=0.8*H_max assumed; disturbance accumulation is not the binding constraint |
| Maximum speed | -- | -- | -- | **6000 rpm** | Representative design choice | Selected from a 3000-15000 rpm trade |
| Rotor inertia | -- | -- | -- | **0.020 kg*m^2** | H_recommended / Omega_max_recommended | Rounded up from the exact quotient |
| Desaturation threshold | H_on=10.053 N*m*s | H_off=5.027 N*m*s | f_on=0.8, f_off=0.4 (unchanged) | -- | Validated against maneuver headroom (M5 sec. 12) | No revision required |
| Dump repeat interval | 779.3 orbits | -- | -- | **51.2 days** | Disturbance secular accumulation | Recomputed for the final wheel's H_max |

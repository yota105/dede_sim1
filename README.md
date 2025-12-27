# Splash Event Generator

Generates droplet reimpact events from a simplified shallow-water heightfield plus ballistic droplets. Outputs JSON Lines with time, position, droplet size, impact speed, estimated pitch, and a relative amplitude proxy.

## Setup

1. Use the provided virtualenv: `D:/制作用/The Dynamism of an Era that is Transmitted and Persists  The Dynamism of the Era that Transmits and Persists/dede_sim1/.venv/Scripts/python.exe`.
2. Install dependencies (already minimal):
   ```powershell
   D:/制作用/The Dynamism of an Era that is Transmitted and Persists  The Dynamism of the Era that Transmits and Persists/dede_sim1/.venv/Scripts/python.exe -m pip install -e .
   ```

## Run

```powershell
D:/制作用/The Dynamism of an Era that is Transmitted and Persists  The Dynamism of the Era that Transmits and Persists/dede_sim1/.venv/Scripts/python.exe src/splash_sim.py --out events.jsonl --n-min 80 --seed 3
```

Optional: override defaults with a JSON file:

```json
{
  "domain": {"domain_size": [2.0, 2.0], "grid_res": [96, 96], "dt": 0.001, "t_end": 1.2, "boundary": "damped"},
  "rock": {"drop_height": 1.2, "rock_radius": 0.1, "noise_amp": 0.08},
  "spawn": {"spawn_rate": 120, "vz_range": [2.0, 6.5], "lateral_noise": 0.6},
  "n_min": 100,
  "seed": 11,
  "output": {"out_path": "events.jsonl"}
}
```

Then run:

```powershell
D:/制作用/The Dynamism of an Era that is Transmitted and Persists  The Dynamism of the Era that Transmits and Persists/dede_sim1/.venv/Scripts/python.exe src/splash_sim.py --config config.json
```

## Output schema (JSONL)

Each line is one event:

```json
{"event_id": 0, "t_hit": 0.153, "x_hit": 1.04, "y_hit": 0.96, "r_drop": 0.0041, "v_hit": 3.2, "f_hz": 3215.8, "amp": 1.7e-7}
```

Fields:
- `event_id`: sequential index.
- `t_hit`: time in seconds.
- `x_hit`, `y_hit`: meters in domain frame.
- `r_drop`: droplet radius in meters.
- `v_hit`: impact normal speed in m/s.
- `f_hz`: estimated bubble resonance frequency using the Minnaert formula $f = \frac{1}{2\pi r_b} \sqrt{\frac{3\gamma P_0}{\rho}}$ with $r_b = c_b r$.
- `amp`: relative loudness proxy $r^3 v^2$.

## Notes

- If `n_min` is not met, the runner doubles `spawn_rate` heuristically and retries once.
- The solver is deliberately simple: forward Euler, open or damped boundaries, Gaussian rock impulse with noise.
- Tune `spawn.spawn_rate`, `rock.drop_height`, and `rock.noise_amp` first when targeting more events.

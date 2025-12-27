"""Minimal hybrid shallow-water + droplet event generator.

Generates droplet reimpact events with estimated pitch (Hz) using a
lightweight heightfield update and ballistic droplet particles.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass
class DomainConfig:
    domain_size: Tuple[float, float] = (2.0, 2.0)  # meters
    grid_res: Tuple[int, int] = (128, 128)
    dt: float = 1e-3
    t_end: float = 1.0
    boundary: str = "open"  # or "damped"
    damping_edge: float = 0.1  # only used when boundary == "damped"


@dataclasses.dataclass
class WaterConfig:
    rho_water: float = 1000.0
    g: float = 9.81
    water_depth0: float = 0.2
    nu_h: float = 0.01  # viscosity-like term for U
    damping: float = 0.1  # velocity damping in the momentum equation


@dataclasses.dataclass
class RockConfig:
    rock_radius: float = 0.1
    rock_mass: float = 5.0
    drop_height: float = 1.0
    drop_velocity0: float = 0.0
    impulse_scale: float = 0.6  # empirical coefficient k_imp
    noise_amp: float = 0.1
    noise_freq: float = 4.0
    center: Tuple[float, float] = (1.0, 1.0)


@dataclasses.dataclass
class SpawnConfig:
    spawn_rate: float = 80.0  # expected droplet count per impulse
    vz_range: Tuple[float, float] = (2.0, 6.0)
    lateral_noise: float = 0.5
    mu_r: float = -5.3  # lognormal mu
    sigma_r: float = 0.35  # lognormal sigma
    max_particles: int = 2000


@dataclasses.dataclass
class DropletPhysics:
    drag_k: float = 0.8  # simple exponential drag coefficient
    ttl: float = 2.0  # seconds until forced removal


@dataclasses.dataclass
class PitchConfig:
    bubble_scale: float = 0.8  # c_b coefficient
    gamma: float = 1.4
    p0: float = 101325.0
    alpha_v: float = 0.0
    beta_v: float = 0.0


@dataclasses.dataclass
class OutputConfig:
    out_path: pathlib.Path = pathlib.Path("events.jsonl")
    log_every: int = 100  # steps between progress logs


@dataclasses.dataclass
class SimConfig:
    domain: DomainConfig = DomainConfig()
    water: WaterConfig = WaterConfig()
    rock: RockConfig = RockConfig()
    spawn: SpawnConfig = SpawnConfig()
    droplet: DropletPhysics = DropletPhysics()
    pitch: PitchConfig = PitchConfig()
    output: OutputConfig = OutputConfig()
    n_min: int = 50  # target minimum event count
    seed: int = 7


@dataclasses.dataclass
class Droplet:
    pos: np.ndarray  # shape (3,)
    vel: np.ndarray  # shape (3,)
    r: float
    born_at: float
    active: bool
    id: int


@dataclasses.dataclass
class Event:
    event_id: int
    t_hit: float
    x_hit: float
    y_hit: float
    r_drop: float
    v_hit: float
    f_hz: float
    amp: float


class HeightField:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        nx, ny = cfg.domain.grid_res
        self.dx = cfg.domain.domain_size[0] / nx
        self.dy = cfg.domain.domain_size[1] / ny
        self.h = np.zeros((nx, ny), dtype=np.float32)  # deviation from H0
        self.u = np.zeros_like(self.h)
        self.v = np.zeros_like(self.h)

    def apply_impulse(self, rng: np.random.Generator) -> None:
        cfg = self.cfg
        rock = cfg.rock
        x0, y0 = rock.center
        nx, ny = cfg.domain.grid_res
        xs = (np.arange(nx) + 0.5) * self.dx
        ys = (np.arange(ny) + 0.5) * self.dy
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        dist = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        r_eff = rock.rock_radius

        v_imp = math.sqrt(max(0.0, rock.drop_velocity0**2 + 2 * cfg.water.g * rock.drop_height))
        area = math.pi * r_eff * r_eff
        base = rock.impulse_scale * rock.rock_mass * v_imp / max(area, 1e-6)

        noise = rock.noise_amp * rng.normal(size=dist.shape)
        kernel = np.exp(-((dist / r_eff) ** 2)) * (1.0 + noise)
        impulse = base * kernel

        # Deposit impulse into velocity as downward push represented by h change.
        self.h -= impulse * 1e-4  # scale down to keep stability

    def step(self, dt: float) -> None:
        cfg = self.cfg
        g = cfg.water.g
        damping = cfg.water.damping
        nu_h = cfg.water.nu_h

        dhdx = self._grad(self.h, axis=0, dx=self.dx)
        dhdy = self._grad(self.h, axis=1, dx=self.dy)
        # Update velocities
        self.u -= dt * (g * dhdx + damping * self.u) + dt * nu_h * self._laplacian(self.u, self.dx, self.dy)
        self.v -= dt * (g * dhdy + damping * self.v) + dt * nu_h * self._laplacian(self.v, self.dx, self.dy)

        # Update height
        div_hu = self._divergence(self.h * self.u, self.h * self.v, self.dx, self.dy)
        self.h -= dt * div_hu

        if self.cfg.domain.boundary == "damped":
            self._apply_edge_damping()

    def surface_height(self, x: float, y: float, H0: float) -> float:
        h = self._bilinear(self.h, x, y) if self._inside(x, y) else 0.0
        return H0 + h

    def _inside(self, x: float, y: float) -> bool:
        Lx, Ly = self.cfg.domain.domain_size
        return 0.0 <= x < Lx and 0.0 <= y < Ly

    def _bilinear(self, field: np.ndarray, x: float, y: float) -> float:
        nx, ny = field.shape
        fx = x / self.dx - 0.5
        fy = y / self.dy - 0.5
        i0 = int(np.floor(fx))
        j0 = int(np.floor(fy))
        i1 = min(i0 + 1, nx - 1)
        j1 = min(j0 + 1, ny - 1)
        sx = fx - i0
        sy = fy - j0
        i0 = np.clip(i0, 0, nx - 1)
        j0 = np.clip(j0, 0, ny - 1)
        v00 = field[i0, j0]
        v10 = field[i1, j0]
        v01 = field[i0, j1]
        v11 = field[i1, j1]
        return (1 - sx) * (1 - sy) * v00 + sx * (1 - sy) * v10 + (1 - sx) * sy * v01 + sx * sy * v11

    def _grad(self, field: np.ndarray, axis: int, dx: float) -> np.ndarray:
        out = np.zeros_like(field)
        if axis == 0:
            out[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dx)
            out[0, :] = (field[1, :] - field[0, :]) / dx
            out[-1, :] = (field[-1, :] - field[-2, :]) / dx
        else:
            out[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
            out[:, 0] = (field[:, 1] - field[:, 0]) / dx
            out[:, -1] = (field[:, -1] - field[:, -2]) / dx
        return out

    def _laplacian(self, field: np.ndarray, dx: float, dy: float) -> np.ndarray:
        out = np.zeros_like(field)
        out[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dx * dx)
            + (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dy * dy)
        )
        return out

    def _divergence(self, fx: np.ndarray, fy: np.ndarray, dx: float, dy: float) -> np.ndarray:
        out = np.zeros_like(fx)
        out[1:-1, 1:-1] = (fx[2:, 1:-1] - fx[:-2, 1:-1]) / (2 * dx) + (fy[1:-1, 2:] - fy[1:-1, :-2]) / (2 * dy)
        out[0, :] = fx[1, :] - fx[0, :]
        out[-1, :] = fx[-1, :] - fx[-2, :]
        out[:, 0] = fy[:, 1] - fy[:, 0]
        out[:, -1] = fy[:, -1] - fy[:, -2]
        return out

    def _apply_edge_damping(self) -> None:
        edge = self.cfg.domain.damping_edge
        nx, ny = self.h.shape
        ramp_x = np.minimum(np.arange(nx), np.arange(nx)[::-1]) / max(nx - 1, 1)
        ramp_y = np.minimum(np.arange(ny), np.arange(ny)[::-1]) / max(ny - 1, 1)
        mask_x = np.clip((edge - ramp_x) / edge, 0.0, 1.0)
        mask_y = np.clip((edge - ramp_y) / edge, 0.0, 1.0)
        mask = np.maximum(mask_x[:, None], mask_y[None, :])
        self.h *= 1.0 - mask
        self.u *= 1.0 - mask
        self.v *= 1.0 - mask


class Simulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.height = HeightField(cfg)
        self.droplets: List[Droplet] = []
        self.events: List[Event] = []
        self.next_id = 0

    def run(self) -> List[Event]:
        cfg = self.cfg
        dt = cfg.domain.dt
        self.height.apply_impulse(self.rng)
        self._spawn_droplets_at_impulse()

        t = 0.0
        step = 0
        while t < cfg.domain.t_end:
            self.height.step(dt)
            self._update_droplets(dt, t)
            t += dt
            step += 1

        return self.events

    def _spawn_droplets_at_impulse(self) -> None:
        cfg = self.cfg
        spawn = cfg.spawn
        count = self.rng.poisson(spawn.spawn_rate)
        for _ in range(int(count)):
            if len(self.droplets) >= spawn.max_particles:
                break
            r = math.exp(self.rng.normal(spawn.mu_r, spawn.sigma_r))
            x0, y0 = cfg.rock.center
            pos = np.array([x0, y0, cfg.water.water_depth0], dtype=np.float32)
            vz = self.rng.uniform(spawn.vz_range[0], spawn.vz_range[1])
            dir_xy = self.rng.normal(scale=spawn.lateral_noise, size=2)
            vel = np.array([dir_xy[0], dir_xy[1], vz], dtype=np.float32)
            self.droplets.append(Droplet(pos=pos.copy(), vel=vel, r=r, born_at=0.0, active=True, id=self.next_id))
            self.next_id += 1

    def _update_droplets(self, dt: float, t: float) -> None:
        cfg = self.cfg
        g = cfg.water.g
        drag_k = cfg.droplet.drag_k
        H0 = cfg.water.water_depth0
        alive: List[Droplet] = []
        for d in self.droplets:
            if not d.active:
                continue
            if t - d.born_at > cfg.droplet.ttl:
                continue
            z_prev = float(d.pos[2])
            h_prev = self.height.surface_height(float(d.pos[0]), float(d.pos[1]), H0)

            d.vel[2] -= g * dt
            d.vel *= math.exp(-drag_k * dt)
            d.pos += d.vel * dt

            x, y, z_now = float(d.pos[0]), float(d.pos[1]), float(d.pos[2])
            if not self.height._inside(x, y):
                continue
            h_now = self.height.surface_height(x, y, H0)

            if z_prev > h_prev and z_now <= h_now and d.vel[2] < 0.0:
                frac = (z_prev - h_prev) / max((z_prev - h_prev) - (z_now - h_now), 1e-6)
                t_hit = t + frac * dt
                x_hit = x - d.vel[0] * (1 - frac) * dt
                y_hit = y - d.vel[1] * (1 - frac) * dt
                v_hit = float(abs(d.vel[2]))
                f_hz = self._minnaert_freq(d.r, v_hit)
                amp = d.r**3 * v_hit * v_hit
                self.events.append(
                    Event(
                        event_id=len(self.events),
                        t_hit=t_hit,
                        x_hit=x_hit,
                        y_hit=y_hit,
                        r_drop=d.r,
                        v_hit=v_hit,
                        f_hz=f_hz,
                        amp=amp,
                    )
                )
                continue
            alive.append(d)
        self.droplets = alive

    def _minnaert_freq(self, r_drop: float, v_hit: float) -> float:
        p = self.cfg.pitch
        rho = self.cfg.water.rho_water
        r_b = max(1e-6, p.bubble_scale * r_drop)
        f0 = (1.0 / (2 * math.pi * r_b)) * math.sqrt((3 * p.gamma * p.p0) / rho)
        return f0 * (1.0 + p.alpha_v * abs(v_hit) + p.beta_v * v_hit * v_hit)


def default_config() -> SimConfig:
    return SimConfig()


def load_config(path: pathlib.Path) -> SimConfig:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return _config_from_dict(raw)


def _config_from_dict(raw: Dict) -> SimConfig:
    def merge(dc_cls, data):
        base = dc_cls()
        for k, v in data.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    cfg = SimConfig()
    if "domain" in raw:
        cfg.domain = merge(DomainConfig, raw["domain"])
    if "water" in raw:
        cfg.water = merge(WaterConfig, raw["water"])
    if "rock" in raw:
        cfg.rock = merge(RockConfig, raw["rock"])
    if "spawn" in raw:
        cfg.spawn = merge(SpawnConfig, raw["spawn"])
    if "droplet" in raw:
        cfg.droplet = merge(DropletPhysics, raw["droplet"])
    if "pitch" in raw:
        cfg.pitch = merge(PitchConfig, raw["pitch"])
    if "output" in raw:
        out = merge(OutputConfig, raw["output"])
        cfg.output = out
    if "n_min" in raw:
        cfg.n_min = raw["n_min"]
    if "seed" in raw:
        cfg.seed = raw["seed"]
    return cfg


def save_events(events: Iterable[Event], path: pathlib.Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(dataclasses.asdict(ev)) + "\n")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate droplet reimpact events for splash pitch estimation.")
    parser.add_argument("--config", type=pathlib.Path, help="Path to JSON config overriding defaults", default=None)
    parser.add_argument("--out", type=pathlib.Path, help="Path to JSONL output", default=None)
    parser.add_argument("--n-min", type=int, help="Minimum desired event count", default=None)
    parser.add_argument("--seed", type=int, help="RNG seed", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else default_config()
    if args.out:
        cfg.output.out_path = args.out
    if args.n_min is not None:
        cfg.n_min = args.n_min
    if args.seed is not None:
        cfg.seed = args.seed

    sim = Simulator(cfg)
    events = sim.run()

    # Optional simple retry by boosting spawn_rate if below target.
    if len(events) < cfg.n_min:
        factor = max(1.5, cfg.n_min / max(len(events), 1))
        cfg.spawn.spawn_rate *= factor
        sim = Simulator(cfg)
        events = sim.run()

    save_events(events, cfg.output.out_path)
    print(f"Generated {len(events)} events -> {cfg.output.out_path}")


if __name__ == "__main__":
    main()

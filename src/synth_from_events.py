"""Synthesize audio from splash event JSONL files.

Generates WAVs using simple sine bursts with exponential decay.
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Dict

import numpy as np
from scipy.io import wavfile

SR = 48_000
BURST_LEN = 0.06  # seconds
ATTACK = 0.003  # seconds
DECAY_TAU = 0.03  # seconds for exponential decay
PEAK_TARGET = 0.9  # final peak normalization target
MASTER_WINDOW = None  # disable global window; rely on per-event envelopes
NOISE_BURST_LEN = 0.01  # seconds of noise layered per event


def load_events(path: pathlib.Path) -> List[Dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def envelope(num_samples: int, sr: int) -> np.ndarray:
    t = np.arange(num_samples) / sr
    env = np.exp(-t / DECAY_TAU)
    # linear attack ramp
    attack_n = max(1, int(ATTACK * sr))
    attack = np.linspace(0.0, 1.0, attack_n, endpoint=True)
    env[:attack_n] *= attack
    return env.astype(np.float32)


def synthesize(events: List[Dict], out_path: pathlib.Path) -> None:
    if not events:
        print(f"No events to synthesize for {out_path}")
        return

    t_max = max(ev["t_hit"] for ev in events) + BURST_LEN
    total_samples = int(np.ceil(t_max * SR))
    audio = np.zeros(total_samples, dtype=np.float32)

    amps = np.array([ev.get("amp", 0.0) for ev in events], dtype=np.float64)
    # Fallback if amp is missing or all zeros: use r_drop^3 * v_hit^2
    if not np.any(amps):
        amps = np.array([ev.get("r_drop", 0.0) ** 3 * ev.get("v_hit", 0.0) ** 2 for ev in events], dtype=np.float64)

    # Robust normalization of per-event gain
    p95 = np.percentile(amps, 95) if np.any(amps) else 1.0
    p95 = max(p95, 1e-9)

    burst_n = int(BURST_LEN * SR)
    env = envelope(burst_n, SR)

    noise_n = int(NOISE_BURST_LEN * SR)
    noise_env = envelope(noise_n, SR)

    for ev, amp_raw in zip(events, amps):
        f = float(ev.get("f_hz", 440.0))
        t0 = int(ev.get("t_hit", 0.0) * SR)
        if t0 >= total_samples:
            continue
        gain = min(1.0, float(amp_raw) / p95) ** 0.5  # mild compression

        t_arr = np.arange(burst_n) / SR
        burst = np.sin(2 * np.pi * f * t_arr).astype(np.float32)
        harmonic = np.sin(2 * np.pi * (2.0 * f) * t_arr).astype(np.float32) * 0.35  # -9 dB
        burst = (burst + harmonic) * env * gain

        if noise_n > 0:
            n = np.random.normal(scale=0.2, size=noise_n).astype(np.float32)
            n *= noise_env
        else:
            n = None

        end = min(total_samples, t0 + burst_n)
        seg_len = end - t0
        if seg_len <= 0:
            continue
        audio[t0:end] += burst[:seg_len]
        if n is not None:
            n_end = min(total_samples, t0 + noise_n)
            n_seg = n_end - t0
            if n_seg > 0:
                audio[t0:n_end] += n[:n_seg]

    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * PEAK_TARGET

    if MASTER_WINDOW == "hann" and total_samples > 4:
        win = np.hanning(total_samples)
        audio *= win.astype(np.float32)
        # renormalize after window so we still hit PEAK_TARGET
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * PEAK_TARGET

    try:
        wavfile.write(out_path, SR, audio)
        print(f"Wrote {out_path} (events={len(events)}, peak={peak:.3f})")
    except PermissionError:
        alt = out_path.with_stem(out_path.stem + "_new")
        wavfile.write(alt, SR, audio)
        print(f"Wrote {alt} (fallback due to permission), events={len(events)}, peak={peak:.3f}")


def main() -> None:
    for src, name in [("events.jsonl", "events_all.wav"), ("events_top100.jsonl", "events_top100.wav")]:
        path = pathlib.Path(src)
        if not path.exists():
            print(f"Skip {src}: not found")
            continue
        events = load_events(path)
        out = pathlib.Path(name)
        synthesize(events, out)


if __name__ == "__main__":
    main()

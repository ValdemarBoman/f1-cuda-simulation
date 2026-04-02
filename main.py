#!/usr/bin/env python3
"""
Simple F1 simulation skeleton - Python version with CUDA support
Optimized for Spark DGX
"""

import csv
import math
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from sac import SAC


@dataclass
class Vec2:
    x: float
    y: float

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float):
        return Vec2(self.x * s, self.y * s)

    def __truediv__(self, s: float):
        return Vec2(self.x / s, self.y / s)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __isub__(self, other):
        self.x -= other.x
        self.y -= other.y
        return self

    def __imul__(self, s: float):
        self.x *= s
        self.y *= s
        return self

    def normalize(self):
        length = math.sqrt(self.x * self.x + self.y * self.y)
        if length > 0:
            self.x /= length
            self.y /= length
        return self

    def normalized(self):
        length = math.sqrt(self.x * self.x + self.y * self.y)
        if length > 0:
            return Vec2(self.x / length, self.y / length)
        return Vec2(0, 0)

    def norm(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def rotate90(self):
        return Vec2(self.y, -self.x)

    def dot(self, other):
        return self.x * other.x + self.y * other.y


@dataclass
class CarParams:
    mass: float
    hp: float


@dataclass
class Physics:
    g: float = 9.82
    air_density: float = 1.225
    lift_coefficient: float = 5


@dataclass
class TrackPoint:
    ppos: Vec2
    wl: float
    wr: float


@dataclass
class TireState:
    lat_fric_coeff: float = 2.0
    long_fric_coeff: float = 1.7


@dataclass
class CarState:
    pos: Vec2
    vel: Vec2
    acc: Vec2
    tire: TireState = field(default_factory=TireState)


@dataclass
class Track:
    points: List[TrackPoint] = field(default_factory=list)
    length: float = 0.0
    T: List[Vec2] = field(default_factory=list)
    N: List[Vec2] = field(default_factory=list)


def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def write_track_csv(tr: Track, path: str):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'x', 'y', 'wl', 'wr'])
        for i, p in enumerate(tr.points):
            writer.writerow([i, p.ppos.x, p.ppos.y, p.wl, p.wr])


def load_track(path: str) -> List[TrackPoint]:
    track = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                p = TrackPoint(
                    ppos=Vec2(float(row[0]), float(row[1])),
                    wr=float(row[2]),
                    wl=float(row[3])
                )
                track.append(p)
            except (ValueError, IndexError):
                continue
    return track


def get_track_length(track: List[TrackPoint]) -> float:
    length = 0.0
    for i in range(1, len(track)):
        dx = track[i].ppos.x - track[i-1].ppos.x
        dy = track[i].ppos.y - track[i-1].ppos.y
        length += math.hypot(dx, dy)
    return length


def nearest_segment_idx(tr: Track, pos: Vec2) -> int:
    best = 0
    best_d2 = 1e100

    n_seg = len(tr.points) - 1
    if n_seg <= 0:
        return 0

    for i in range(n_seg):
        a = tr.points[i].ppos
        b = tr.points[i+1].ppos
        ab = b - a
        ab2 = ab.dot(ab)
        if ab2 < 1e-9:
            continue

        ap = pos - a
        t = ap.dot(ab) / ab2
        t = max(0.0, min(1.0, t))

        proj = a + ab * t
        d = pos - proj
        d2 = d.dot(d)

        if d2 < best_d2:
            best_d2 = d2
            best = i

    return best


def build_arc_length_s(tr: Track) -> List[float]:
    S = [0.0] * len(tr.points)
    for i in range(1, len(tr.points)):
        dx = tr.points[i].ppos.x - tr.points[i-1].ppos.x
        dy = tr.points[i].ppos.y - tr.points[i-1].ppos.y
        S[i] = S[i-1] + math.hypot(dx, dy)
    return S


def get_track_tangents(track: List[TrackPoint]) -> List[Vec2]:
    tangents = []
    for i in range(1, len(track)):
        dir_vec = track[i].ppos - track[i-1].ppos
        dir_vec.normalize()
        tangents.append(dir_vec)
    return tangents


def get_track_normals(tangents: List[Vec2]) -> List[Vec2]:
    return [t.rotate90() for t in tangents]


def clampd(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_normal_force(car: CarState, phy: Physics, car_p: CarParams) -> float:
    weight = phy.g * car_p.mass
    v_norm = car.vel.norm()
    downforce = 0.5 * phy.air_density * phy.lift_coefficient * v_norm * v_norm
    return weight + downforce


def get_observations(s: float, offset: float, curvature: float, lookahead: float,
                     v_dir: float, v_perp: float, a_dir_prev: float, a_perp_prev: float,
                     track_length: float, max_offset: float) -> List[float]:
    obs = [
        s / track_length,
        offset / max_offset,
        curvature / 0.05,
        lookahead / 0.05,
        v_dir / 100.0,
        v_perp / 50.0,
        a_dir_prev,
        a_perp_prev,
    ]
    return obs


def get_curvature_at_idx(idx: int, tr: Track) -> float:
    n_t = len(tr.T)
    n_p = len(tr.points)
    if n_t < 2 or n_p < 3:
        return 0.0

    idx = max(0, min(idx, n_t - 2))

    t0 = tr.T[idx]
    t1 = tr.T[idx + 1]

    dotv = max(-1.0, min(1.0, t0.dot(t1)))
    cross = t0.x * t1.y - t0.y * t1.x

    dtheta = math.atan2(cross, dotv)

    p0 = tr.points[idx].ppos
    p1 = tr.points[idx + 1].ppos
    ds = math.hypot(p1.x - p0.x, p1.y - p0.y)

    if ds < 1e-3:
        return 0.0

    return dtheta / ds


def get_track_index(pos: Vec2, tr: Track) -> int:
    min_dist = 1e9
    closest_idx = 0
    for i, p in enumerate(tr.points):
        dist = math.hypot(pos.x - p.ppos.x, pos.y - p.ppos.y)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    return closest_idx


def get_track_position(pos: Vec2, tr: Track, S: List[float]) -> float:
    i = nearest_segment_idx(tr, pos)

    a = tr.points[i].ppos
    b = tr.points[i+1].ppos
    ab = b - a
    seg_len = math.hypot(ab.x, ab.y)
    if seg_len < 1e-9:
        return S[i]

    ab2 = ab.dot(ab)
    ap = pos - a
    t = ap.dot(ab) / ab2
    t = max(0.0, min(1.0, t))

    return S[i] + t * seg_len


def get_offset(pos: Vec2, tr: Track) -> float:
    i = nearest_segment_idx(tr, pos)

    a = tr.points[i].ppos
    b = tr.points[i+1].ppos
    ab = b - a
    ab2 = ab.dot(ab)
    if ab2 < 1e-9:
        return 0.0

    ap = pos - a
    t = ap.dot(ab) / ab2
    t = max(0.0, min(1.0, t))

    proj = a + ab * t

    n = tr.N[i]
    d = pos - proj
    return d.dot(n)


def progress_ds(s_prev: float, s_now: float, L: float) -> float:
    ds = s_now - s_prev
    if ds < -0.5 * L:
        ds += L
    if ds > 0.5 * L:
        ds -= L
    return ds


def off_track_from_offset(tr: Track, seg_idx: int, offset_signed: float) -> Tuple[float, float]:
    pt_idx = max(0, min(seg_idx, len(tr.points) - 1))
    left = tr.points[pt_idx].wl
    right = tr.points[pt_idx].wr
    return left, right


def main():
    tr = Track()
    car_p = CarParams(mass=800, hp=1000)
    car = CarState(pos=Vec2(0, 0), vel=Vec2(0, 0), acc=Vec2(0, 0))
    phy = Physics()

    tr.points = load_track("Silverstone.csv")
    S = build_arc_length_s(tr)
    tr.length = get_track_length(tr.points)
    tr.T = get_track_tangents(tr.points)
    tr.N = get_track_normals(tr.T)

    ensure_dir("logs")
    write_track_csv(tr, "logs/track_centerline.csv")

    step_log = open("logs/metrics_step.csv", 'w', newline='')
    ep_log = open("logs/metrics_episode.csv", 'w', newline='')

    step_writer = csv.writer(step_log)
    ep_writer = csv.writer(ep_log)

    step_writer.writerow([
        "episode", "step", "t", "s", "lapProgress", "ds", "offset", "leftWidth", "rightWidth",
        "off", "finish", "x", "y", "vx", "vy", "speed", "v_dir", "v_perp",
        "ax_cmd_norm", "ay_cmd_norm", "ax_local", "ay_local", "ax_g", "ay_g",
        "curvature", "curv_lookahead", "N",
        "reward", "ds_reward", "v_dir_reward", "brake_penalty",
        "pen_action", "pen_offset", "pen_vperp", "pen_curv", "pen_curvLA", "term_off", "term_finish"
    ])

    ep_writer.writerow(["episode", "steps", "epReturn", "lapProgress", "dist", "done_off", "done_finish", "replaySize"])

    OBS_DIM = 8
    sac = SAC(OBS_DIM, 2, 1234)
    sac.learn_start = 2000
    sac.batch_size = 128
    sac.updates_per_step = 1
    sac.alpha = 0.25

    EPISODES = 100
    MAX_STEPS = 6000

    progresses = []

    for ep in range(EPISODES):
        ep_return = 0.0
        car.pos = Vec2(tr.points[0].ppos.x, tr.points[0].ppos.y)
        car.vel = Vec2(0.0, 0.0)
        car.acc = Vec2(0.0, 0.0)

        dt = 0.05
        t = 0.0
        s = 0.0
        s_prev = get_track_position(car.pos, tr, S)
        lap_progress = 0.0
        armed_finish = False
        dist = 0.0

        traj_log = open(f"logs/episode_{ep}_traj.csv", 'w', newline='')
        traj_writer = csv.writer(traj_log)
        traj_writer.writerow(["episode", "step", "t", "x", "y", "s", "lapProgress", "offset"])

        prev_obs = get_observations(s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tr.length, 10.0)
        assert len(prev_obs) == OBS_DIM
        prev_act = [0.2, 0.0]

        i = 0
        while s < tr.length:
            s_before = get_track_position(car.pos, tr, S)

            seg0 = nearest_segment_idx(tr, car.pos)
            seg0 = max(0, min(seg0, len(tr.T) - 1))

            dir0 = tr.T[seg0]
            right0 = tr.N[seg0]

            N = get_normal_force(car, phy, car_p)

            a_dir_griplimit = (car.tire.long_fric_coeff * N) / car_p.mass
            a_perp_griplimit = (car.tire.lat_fric_coeff * N) / car_p.mass

            car.acc.x = prev_act[0] * a_dir_griplimit if a_dir_griplimit > 0 else 0
            car.acc.y = prev_act[1] * a_perp_griplimit if a_perp_griplimit > 0 else 0
            car.acc.x = clampd(car.acc.x, -abs(a_dir_griplimit), abs(a_dir_griplimit))
            car.acc.y = clampd(car.acc.y, -abs(a_perp_griplimit), abs(a_perp_griplimit))

            nx = car.acc.x / a_dir_griplimit if abs(a_dir_griplimit) > 1e-6 else 0
            ny = car.acc.y / a_perp_griplimit if abs(a_perp_griplimit) > 1e-6 else 0

            q = nx*nx + ny*ny
            if q > 1.0:
                s_sq = 1.0 / math.sqrt(max(q, 1e-6))
                car.acc.x *= s_sq
                car.acc.y *= s_sq

            acc_dir_gs = car.acc.x / 9.82
            acc_perp_gs = car.acc.y / 9.82

            speed = max(car.vel.norm(), 0.1)
            a_dir_powerlimit = (car_p.hp * 745.7) / (speed * car_p.mass) if speed > 0 else 1e6
            if car.acc.x > a_dir_powerlimit:
                car.acc.x = a_dir_powerlimit

            a_global = dir0 * car.acc.x + right0 * car.acc.y

            car.vel += a_global * dt
            car.pos += car.vel * dt
            dist += car.vel.norm() * dt

            s_after = get_track_position(car.pos, tr, S)

            seg1 = nearest_segment_idx(tr, car.pos)
            seg1 = max(0, min(seg1, len(tr.T) - 1))

            dir1 = tr.T[seg1]
            right1 = tr.N[seg1]

            offset = get_offset(car.pos, tr)

            ds = progress_ds(s_prev, s_after, tr.length)
            s_prev = s_after

            lap_progress += max(0.0, ds)

            if not armed_finish and lap_progress > 5000:
                armed_finish = True

            lookahead_pts = 15
            seg_curv = min(seg1, len(tr.T) - 2)
            seg_la = min(seg_curv + lookahead_pts, len(tr.T) - 2)

            curvature = get_curvature_at_idx(seg_curv, tr)
            curvature_lookahead = get_curvature_at_idx(seg_la, tr)

            v_dir = car.vel.dot(dir1)
            v_perp = car.vel.dot(right1)

            off = False
            left, right = off_track_from_offset(tr, seg1, -1e-9)
            if offset < -left or offset > right:
                off = True

            done = False
            if off:
                done = True
            if not (math.isfinite(car.pos.x) and math.isfinite(car.pos.y) and
                    math.isfinite(car.vel.x) and math.isfinite(car.vel.y)):
                done = True

            finish = False
            if armed_finish and lap_progress >= tr.length:
                done = True
                finish = True

            ds_reward = ds

            v_fwd = max(0.0, v_dir)
            v_rev = max(0.0, -v_dir)

            k_now = curvature
            k_la = curvature_lookahead
            k_now_abs = abs(k_now)
            k_la_abs = abs(k_la)

            gate_now = clampd(k_now_abs / 0.02, 0.0, 1.0)
            gate_la = clampd(k_la_abs / 0.02, 0.0, 1.0)
            pre_gate = gate_la * (1.0 - gate_now)
            in_gate = gate_now

            a_lat_req = v_fwd * v_fwd * k_la_abs
            a_lat_max = 0.85 * a_perp_griplimit
            over_a = max(0.0, a_lat_req - a_lat_max)

            pen_overspeed = -0.0020 * over_a * over_a
            pen_overspeed = max(pen_overspeed, -3.0)

            brake = max(0.0, -prev_act[0])
            brake_bonus = (pre_gate + 0.5*in_gate) * 0.25 * brake * min(over_a, 10.0)

            half_width = max(1e-3, 0.5 * (left + right))
            offset_norm = clampd(offset / half_width, -1.0, 1.0)

            desired_outside_sign = 1.0 if k_la >= 0.0 else -1.0
            desired_inside_sign = -desired_outside_sign

            align_outside = desired_outside_sign * offset_norm
            align_inside = desired_inside_sign * offset_norm

            bonus_outside_pre = pre_gate * 0.12 * max(0.0, align_outside)
            pen_inside_pre = pre_gate * -0.04 * max(0.0, -align_outside)

            bonus_inside_in = in_gate * 0.08 * max(0.0, align_inside)
            pen_outside_in = in_gate * -0.02 * max(0.0, -align_inside)

            w_offset = (1.0 - gate_la) * 0.15 + gate_la * 0.03
            pen_offset = -w_offset * abs(offset)

            pen_vperp = -0.25 * abs(v_perp)

            pen_action = -0.05 * (prev_act[0]*prev_act[0] + prev_act[1]*prev_act[1])

            pen_reverse = -1.0 * v_rev

            v_dir_reward = 0.002 * v_fwd

            term_off = -8.0 if off else 0.0
            term_finish = 100.0 if finish else 0.0

            reward = (
                4.0 * ds_reward
                + v_dir_reward
                + pen_action
                + pen_offset
                + pen_vperp
                + pen_reverse
                + pen_overspeed
                + brake_bonus
                + bonus_outside_pre + pen_inside_pre
                + bonus_inside_in + pen_outside_in
                + term_off
                + term_finish
            )

            s = s_after

            observations = get_observations(s, offset, curvature, curvature_lookahead,
                                          v_dir, v_perp, prev_act[0], prev_act[1], tr.length, left + right)

            sac.store(prev_obs, prev_act, reward, observations, done)
            sac.update_many(sac.updates_per_step)

            prev_act = sac.act(observations, False)
            prev_obs = observations

            ep_return += reward

            if lap_progress > 75:
                print(f"Step {i} offset={offset:.2f} reward={reward:.4f} step_progress={ds:.4f} "
                      f"progress={lap_progress:.2f} dist={dist:.2f} vel={car.vel.norm():.2f} "
                      f"dir={v_dir:.2f} perp={v_perp:.2f} off={int(off)} time={t:.5f} "
                      f"prev_act=[{prev_act[0]:.4f},{prev_act[1]:.4f}] "
                      f"acc_gs=[{acc_dir_gs:.4f},{acc_perp_gs:.4f}] "
                      f"curvature={curvature:.6f} lookahead_curvature={curvature_lookahead:.6f}\n")

            ax_cmd_norm = prev_act[0]
            ay_cmd_norm = prev_act[1]

            ax_local = car.acc.x
            ay_local = car.acc.y

            ax_g = a_global.x
            ay_g = a_global.y

            pen_action_log = -0.01 * (prev_act[0]*prev_act[0] + prev_act[1]*prev_act[1])
            pen_offset_log = -0.15 * abs(offset)
            pen_vperp_log = -0.5 * abs(v_perp)
            pen_curv_log = -0.02 * (car.vel.norm()*car.vel.norm() * abs(curvature))
            pen_curvLA_log = -0.03 * (car.vel.norm()*car.vel.norm() * abs(curvature_lookahead))
            term_off_log = -10.0 if off else 0.0
            term_fin_log = 100.0 if finish else 0.0

            step_writer.writerow([
                ep, i, t, s_after, lap_progress, ds, offset, left, right,
                int(off), int(finish),
                car.pos.x, car.pos.y, car.vel.x, car.vel.y, car.vel.norm(), v_dir, v_perp,
                ax_cmd_norm, ay_cmd_norm, ax_local, ay_local, ax_g, ay_g,
                curvature, curvature_lookahead, N,
                reward, ds_reward, v_dir_reward, pen_action_log,
                pen_action_log, pen_offset_log, pen_vperp_log, pen_curv_log, pen_curvLA_log, term_off_log, term_fin_log
            ])
            step_log.flush()

            traj_writer.writerow([ep, i, t, car.pos.x, car.pos.y, s_after, lap_progress, offset])
            traj_log.flush()

            if done or i >= MAX_STEPS:
                progresses.append(lap_progress)
                print(f" Time{t:.2f} s={s:.2f} offset={offset:.2f} progress={lap_progress:.2f} "
                      f"Dist={dist:.2f} Episode {ep} return={ep_return:.2f} replay={sac.replay_size()}\n")

                ep_writer.writerow([ep, i, ep_return, lap_progress, dist, int(off), int(finish), sac.replay_size()])
                ep_log.flush()
                break

            t += dt
            i += 1

        traj_log.close()

    step_log.close()
    ep_log.close()


if __name__ == "__main__":
    main()

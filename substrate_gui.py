"""
Substrate GUI — live preview with tkinter
Numba @njit accelerated inner loop (~35x faster than pure Python).
Falls back to pure Python if numba is not installed.
"""

import math
import random
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Numba import
# ---------------------------------------------------------------------------

try:
    from numba import njit
    NUMBA = True
except ImportError:
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return decorator if (args and callable(args[0])) else decorator
    NUMBA = False

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Palettes — all defined before @njit so numba captures them
# ---------------------------------------------------------------------------

PALETTES = {
    "Tarbell Original": np.array([
        [214,178,120],[190,140, 80],[160,100, 50],[220,200,160],
        [180,160,130],[200,180,140],[240,220,180],[160,130, 90],
        [100, 70, 40],[230,210,170],[170,140,100],[210,190,150],
        [140,110, 70],[250,230,190],[120, 90, 55],[195,165,115],
        [175,145, 95],[205,175,125],[185,155,105],[165,135, 85],
        [ 80, 55, 30],[245,225,185],[155,125, 75],[225,205,165],
        [135,105, 65],[215,195,155],[145,115, 70],[235,215,175],
        [125, 95, 60],[255,235,195],[115, 85, 50],[205,185,145],
        [120,140,160],[ 90,110,140],[150,170,190],[ 70, 90,120],
        [100,130,150],[140,160,180],[ 60, 80,110],[130,150,170],
        [160,130,110],[180,150,130],[200,170,150],[140,110, 90],
    ], dtype=np.float64),

    "Vivid": np.array([
        [180,120, 60],[140, 80, 30],[100, 50, 20],[200,150, 80],
        [160,100, 40],[220,170, 90],[120, 70, 25],[240,190,110],
        [ 40, 80,140],[ 20, 60,120],[ 60,100,160],[ 30, 70,130],
        [ 50,120,100],[ 30, 90, 70],[ 70,140,110],[ 40,110, 80],
        [140, 50, 50],[120, 40, 60],[160, 70, 70],[100, 40, 80],
        [150, 60, 90],[130, 45, 55],[170, 80, 60],[110, 50, 70],
        [140,130, 40],[120,110, 30],[160,150, 50],[100, 90, 20],
        [ 40, 40, 60],[ 60, 40, 40],[ 40, 60, 40],[ 50, 50, 70],
        [220,200,180],[200,220,210],[210,200,230],[230,210,190],
        [190,210,220],[215,195,215],[205,220,200],[225,205,185],
        [ 80, 50, 30],[ 30, 50, 80],[ 50, 80, 50],[ 60, 30, 60],
    ], dtype=np.float64),

    "Ocean": np.array([
        [ 10, 60,100],[ 20, 80,130],[ 30,100,150],[ 15, 70,120],
        [ 40,120,160],[ 50,140,170],[ 60,150,180],[ 25, 90,140],
        [ 80,160,190],[100,170,200],[120,180,210],[ 70,150,185],
        [140,190,215],[160,200,220],[180,210,225],[150,195,218],
        [ 10, 40, 80],[ 20, 50, 90],[ 30, 60,100],[ 15, 45, 85],
        [200,220,230],[210,225,235],[220,230,240],[205,222,232],
        [ 60,130,160],[ 70,140,170],[ 80,150,180],[ 65,135,165],
        [  5, 30, 60],[  8, 35, 65],[ 12, 40, 70],[  6, 32, 62],
        [170,205,220],[180,210,225],[190,215,228],[175,207,222],
        [100,160,190],[110,165,195],[120,170,200],[105,162,192],
        [ 90,155,185],[ 95,158,188],[ 85,152,182],[ 88,154,184],
    ], dtype=np.float64),

    "Ember": np.array([
        [180, 40, 10],[160, 30,  8],[200, 60, 15],[140, 25,  5],
        [220, 80, 20],[200, 70, 18],[240,100, 25],[210, 75, 20],
        [240,140, 30],[220,120, 25],[200,100, 20],[230,130, 28],
        [250,180, 50],[240,160, 40],[230,150, 35],[245,170, 45],
        [255,200, 80],[250,190, 70],[245,185, 65],[252,195, 75],
        [255,220,120],[252,210,100],[248,205, 90],[254,215,110],
        [ 80, 20,  5],[ 60, 15,  3],[ 40, 10,  2],[ 70, 18,  4],
        [100, 30, 10],[ 90, 25,  8],[110, 35, 12],[ 95, 28,  9],
        [255,240,180],[252,235,170],[248,230,160],[254,238,175],
        [120, 40, 15],[130, 45, 18],[140, 50, 20],[125, 42, 16],
        [160, 60, 25],[170, 65, 28],[150, 55, 22],[165, 62, 26],
    ], dtype=np.float64),

    "Monochrome": np.array([
        [ 20, 20, 20],[ 30, 30, 30],[ 40, 40, 40],[ 50, 50, 50],
        [ 60, 60, 60],[ 70, 70, 70],[ 80, 80, 80],[ 90, 90, 90],
        [100,100,100],[110,110,110],[120,120,120],[130,130,130],
        [140,140,140],[150,150,150],[160,160,160],[170,170,170],
        [180,180,180],[190,190,190],[200,200,200],[210,210,210],
        [ 25, 25, 25],[ 35, 35, 35],[ 45, 45, 45],[ 55, 55, 55],
        [ 65, 65, 65],[ 75, 75, 75],[ 85, 85, 85],[ 95, 95, 95],
        [105,105,105],[115,115,115],[125,125,125],[135,135,135],
        [145,145,145],[155,155,155],[165,165,165],[175,175,175],
        [185,185,185],[195,195,195],[205,205,205],[215,215,215],
        [ 15, 15, 15],[ 22, 22, 22],[ 48, 48, 48],[ 72, 72, 72],
    ], dtype=np.float64),

    "Forest": np.array([
        [ 30, 60, 20],[ 40, 75, 25],[ 50, 90, 30],[ 35, 68, 22],
        [ 60,100, 35],[ 70,110, 40],[ 80,120, 45],[ 65,105, 38],
        [ 90,130, 50],[100,140, 55],[110,150, 60],[ 95,135, 52],
        [120,160, 65],[130,170, 70],[140,180, 75],[125,165, 68],
        [ 80,100, 40],[ 90,110, 45],[100,120, 50],[ 85,105, 42],
        [160,140, 60],[140,120, 50],[120,100, 40],[150,130, 55],
        [180,160, 80],[160,140, 70],[140,120, 60],[170,150, 75],
        [ 20, 40, 15],[ 25, 50, 18],[ 30, 55, 20],[ 22, 45, 16],
        [200,190,140],[190,180,130],[180,170,120],[195,185,135],
        [ 60, 80, 30],[ 70, 90, 35],[ 50, 70, 25],[ 65, 85, 32],
        [110,130, 55],[120,140, 60],[100,120, 50],[115,135, 58],
    ], dtype=np.float64),
}

PALETTE_NAMES = list(PALETTES.keys())

# Build a single stacked array: shape (N_PALETTES, 44, 3)
# numba will index into this with a palette index
_PAL_SIZE = 44
ALL_PALETTES = np.zeros((len(PALETTE_NAMES), _PAL_SIZE, 3), dtype=np.float64)
for _i, _name in enumerate(PALETTE_NAMES):
    _p = PALETTES[_name]
    _n = min(len(_p), _PAL_SIZE)
    ALL_PALETTES[_i, :_n] = _p[:_n]
    if _n < _PAL_SIZE:   # tile if shorter than 44
        for _j in range(_n, _PAL_SIZE):
            ALL_PALETTES[_i, _j] = _p[_j % _n]

# Keep PALETTE_NB pointing at the default (Vivid) for backward compat
PALETTE_NB = ALL_PALETTES[1]   # "Vivid"
PAL_LEN = _PAL_SIZE
MAX_CRACKS = 2048

# ---------------------------------------------------------------------------
# Numba JIT inner loop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _run_batch(canvas, cgrid, W, H,
               crack_x, crack_y, crack_t,
               sand_c, sand_g,
               rng_state,
               alive,
               batch_size, num_cracks,
               origin_x, origin_y, origin_bias,
               palette_idx):
    PI180 = math.pi / 180.0
    CRACK_KEEP = (255.0 - 85.0) / 255.0
    M1 = np.uint64(6364136223846793005)
    M2 = np.uint64(1442695040888963407)
    MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

    for _ in range(batch_size):
        any_alive = False

        for ci in range(num_cracks):
            if not alive[ci]:
                continue
            any_alive = True

            s = rng_state[ci]
            t = crack_t[ci]
            cos_t = math.cos(t * PI180)
            sin_t = math.sin(t * PI180)

            # advance
            crack_x[ci] += 0.42 * cos_t
            crack_y[ci] += 0.42 * sin_t

            # fuzzed grid coords
            s = (s * M1 + M2) & MASK
            v = float(s >> np.uint64(33)) / 2147483648.0
            cx = int(crack_x[ci] + v * 0.66 - 0.33)
            s = (s * M1 + M2) & MASK
            v = float(s >> np.uint64(33)) / 2147483648.0
            cy = int(crack_y[ci] + v * 0.66 - 0.33)

            # walk perpendicular to find region boundary (max 300px)
            rx = crack_x[ci]
            ry = crack_y[ci]
            for _w in range(300):
                rx += 0.81 * sin_t
                ry -= 0.81 * cos_t
                rcx = int(rx)
                rcy = int(ry)
                if 0 <= rcx < W and 0 <= rcy < H:
                    if cgrid[rcy * W + rcx] <= 10000:
                        break
                else:
                    break

            # sand painter
            s = (s * M1 + M2) & MASK
            v = float(s >> np.uint64(33)) / 2147483648.0
            sand_g[ci] += v * 0.1 - 0.05
            if sand_g[ci] < 0.0: sand_g[ci] = 0.0
            if sand_g[ci] > 1.0: sand_g[ci] = 1.0

            grains = 64
            w_step = sand_g[ci] / float(grains - 1)
            ox = crack_x[ci]
            oy = crack_y[ci]
            sc0 = sand_c[ci, 0]
            sc1 = sand_c[ci, 1]
            sc2 = sand_c[ci, 2]

            for i in range(grains):
                t_val = math.sin(math.sin(float(i) * w_step))
                px = int(ox + (rx - ox) * t_val)
                py = int(oy + (ry - oy) * t_val)
                if 0 <= px < W and 0 <= py < H:
                    alpha = 0.1 - float(i) / float(grains * 10)
                    if alpha > 0.0:
                        inv = 1.0 - alpha
                        canvas[py, px, 0] = int(canvas[py, px, 0] * inv + sc0 * alpha)
                        canvas[py, px, 1] = int(canvas[py, px, 1] * inv + sc1 * alpha)
                        canvas[py, px, 2] = int(canvas[py, px, 2] * inv + sc2 * alpha)

            # crack pixel
            ix = int(crack_x[ci])
            iy = int(crack_y[ci])
            if 0 <= ix < W and 0 <= iy < H:
                canvas[iy, ix, 0] = int(canvas[iy, ix, 0] * CRACK_KEEP)
                canvas[iy, ix, 1] = int(canvas[iy, ix, 1] * CRACK_KEEP)
                canvas[iy, ix, 2] = int(canvas[iy, ix, 2] * CRACK_KEEP)

            # update grid
            needs_restart = False
            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                needs_restart = True
            else:
                idx = cy * W + cx
                existing = cgrid[idx]
                if existing > 10000 or abs(float(existing) - t) < 5.0:
                    cgrid[idx] = int(t)
                elif abs(float(existing) - t) > 2.0:
                    needs_restart = True

            # restart or retire — biased find_start
            # samples up to 200 candidates, picks closest to origin
            # when origin_bias > 0
            if needs_restart:
                found = False
                fpx = 0
                fpy = 0
                best_fpx = 0
                best_fpy = 0
                best_dist = 1e18

                # Biased find_start:
                # - samples up to 40 candidates
                # - scores each by distance to origin (closer = better)
                # - BUT enforces a minimum spread distance so cracks
                #   don't pile up in one pixel near the origin
                # n_candidates: slider 0-100% → 1-3 candidates
                # 3 is the safe maximum before dense-area loops occur
                n_candidates = 1 + int((origin_bias / 0.04) * 2.0)
                for _t in range(200):
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    fpx = int(v * W)
                    if fpx >= W: fpx = W - 1
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    fpy = int(v * H)
                    if fpy >= H: fpy = H - 1
                    if cgrid[fpy * W + fpx] < 10000:
                        dx = float(fpx) - origin_x
                        dy = float(fpy) - origin_y
                        dist = dx*dx + dy*dy
                        if dist < best_dist:
                            best_dist = dist
                            best_fpx = fpx
                            best_fpy = fpy
                        found = True
                        n_candidates -= 1
                        if n_candidates <= 0:
                            break

                if not found:
                    alive[ci] = 0
                else:
                    fpx = best_fpx
                    fpy = best_fpy
                    fa = float(cgrid[fpy * W + fpx])
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    sign = 1.0 if v < 0.5 else -1.0
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    fa += sign * 90.0 + (v * 4.0 - 2.0)
                    crack_x[ci] = float(fpx) + 0.61 * math.cos(fa * PI180)
                    crack_y[ci] = float(fpy) + 0.61 * math.sin(fa * PI180)
                    crack_t[ci] = fa
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    pi = int(v * PAL_LEN)
                    if pi >= PAL_LEN: pi = PAL_LEN - 1
                    sand_c[ci, 0] = ALL_PALETTES[palette_idx, pi, 0]
                    sand_c[ci, 1] = ALL_PALETTES[palette_idx, pi, 1]
                    sand_c[ci, 2] = ALL_PALETTES[palette_idx, pi, 2]
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    sand_g[ci] = 0.01 + v * 0.09

            rng_state[ci] = s

        if not any_alive:
            break

    # report how many cracks are still alive
    return alive


# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

class _SandPainter:
    def __init__(self, rng):
        self.c = PALETTE_NB[rng.integers(0, PAL_LEN)].copy()
        self.g = rng.uniform(0.01, 0.1)

    def render(self, canvas, rx, ry, ox, oy):
        self.g = float(np.clip(self.g + np.random.uniform(-0.05, 0.05), 0, 1))
        H, W = canvas.shape[:2]
        grains = 64
        w = self.g / (grains - 1)
        for i in range(grains):
            t = math.sin(math.sin(i * w))
            px = int(ox + (rx - ox) * t)
            py = int(oy + (ry - oy) * t)
            if 0 <= px < W and 0 <= py < H:
                alpha = max(0.0, 0.1 - i / (grains * 10.0))
                canvas[py, px] = np.clip(
                    canvas[py, px] * (1 - alpha) + self.c * alpha, 0, 255
                ).astype(np.uint8)


class _Crack:
    PI180 = math.pi / 180.0

    def __init__(self, W, H, cgrid, rng):
        self.W, self.H, self.cgrid, self.rng = W, H, cgrid, rng
        self.x = self.y = self.t = 0.0
        self.sp = _SandPainter(rng)
        self.alive = True
        self._find_start()

    def _find_start(self):
        found = False
        px = py = 0
        for _ in range(1000):
            px = int(self.rng.integers(0, self.W))
            py = int(self.rng.integers(0, self.H))
            if self.cgrid[py * self.W + px] < 10000:
                found = True
                break
        if not found:
            self.alive = False   # grid is full, retire
            return
        a = float(self.cgrid[py * self.W + px])
        a += 90 * int(self.rng.choice([-1, 1])) + int(self.rng.integers(-2, 3))
        self.x = px + 0.61 * math.cos(a * self.PI180)
        self.y = py + 0.61 * math.sin(a * self.PI180)
        self.t = a
        self.sp = _SandPainter(self.rng)

    def move(self, canvas):
        PI180 = self.PI180
        self.x += 0.42 * math.cos(self.t * PI180)
        self.y += 0.42 * math.sin(self.t * PI180)
        cx = int(self.x + self.rng.uniform(-0.33, 0.33))
        cy = int(self.y + self.rng.uniform(-0.33, 0.33))

        # region color
        rx, ry = self.x, self.y
        sin_t = math.sin(self.t * PI180)
        cos_t = math.cos(self.t * PI180)
        for _ in range(1000):
            rx += 0.81 * sin_t
            ry -= 0.81 * cos_t
            rcx, rcy = int(rx), int(ry)
            if 0 <= rcx < self.W and 0 <= rcy < self.H:
                if self.cgrid[rcy * self.W + rcx] <= 10000:
                    break
            else:
                break
        self.sp.render(canvas, rx, ry, self.x, self.y)

        # crack pixel — correct alpha blend
        ix, iy = int(self.x), int(self.y)
        if 0 <= ix < self.W and 0 <= iy < self.H:
            canvas[iy, ix] = (canvas[iy, ix] * ((255 - 85) / 255.0)).astype(np.uint8)

        if cx < 0 or cx >= self.W or cy < 0 or cy >= self.H:
            self._find_start(); return
        idx = cy * self.W + cx
        existing = self.cgrid[idx]
        if existing > 10000 or abs(existing - self.t) < 5:
            self.cgrid[idx] = int(self.t)
        elif abs(existing - self.t) > 2:
            self._find_start()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SubstrateEngine:
    BATCH = 500

    def __init__(self, width, height, num_cracks, seed=None, fill_pct=95,
                 initial_cracks=3, num_seeds=16, vsync=False,
                 origin_x=0.0, origin_y=0.0, origin_bias=0.0,
                 palette_idx=1):
        self.W = width
        self.H = height
        self.num_cracks    = min(num_cracks, MAX_CRACKS)
        self.initial_cracks = max(1, min(initial_cracks, num_cracks))
        self.step = 0
        self.running = False
        self.lock = threading.Lock()
        self.fill_pct = max(1, min(100, fill_pct)) / 100.0
        self.vsync = vsync
        self._frame_consumed = threading.Event()
        self._frame_consumed.set()
        # origin bias: normalised 0-1 coords, bias strength 0-1
        self.origin_x    = float(origin_x) * width
        self.origin_y    = float(origin_y) * height
        self.origin_bias = float(origin_bias)
        self.palette_idx = int(palette_idx)

        rng_seed = seed if seed is not None else random.randint(0, 2**31)
        self.rng = np.random.default_rng(rng_seed)

        self.canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        self.cgrid  = np.full(width * height, 10001, dtype=np.int32)

        # seed initial crack directions
        # origin_bias controls how tightly seeds cluster around origin:
        # 0 = uniform random, 1 = tight gaussian cluster at origin
        n_seeds = max(1, min(num_seeds, 256))
        ox_px = self.origin_x
        oy_px = self.origin_y
        for _ in range(n_seeds):
            if self.origin_bias > 0:
                # sigma: 0% bias = 50% of canvas, 100% bias = 5% of canvas
                bias_norm = self.origin_bias / 0.04
                sigma = max(width, height) * (0.5 - bias_norm * 0.45)
                x = int(np.clip(self.rng.normal(ox_px, sigma), 0, width - 1))
                y = int(np.clip(self.rng.normal(oy_px, sigma), 0, height - 1))
            else:
                x = int(self.rng.integers(0, width))
                y = int(self.rng.integers(0, height))
            self.cgrid[y * width + x] = int(self.rng.integers(0, 360))

        if NUMBA:
            self._init_numba(rng_seed)
        else:
            self._init_python()

    def _init_numba(self, rng_seed):
        self.crack_x   = np.zeros(MAX_CRACKS, dtype=np.float64)
        self.crack_y   = np.zeros(MAX_CRACKS, dtype=np.float64)
        self.crack_t   = np.zeros(MAX_CRACKS, dtype=np.float64)
        self.sand_c    = np.zeros((MAX_CRACKS, 3), dtype=np.float64)
        self.sand_g    = np.full(MAX_CRACKS, 0.05, dtype=np.float64)
        # start with only initial_cracks alive; rest start dead
        self.alive     = np.zeros(MAX_CRACKS, dtype=np.int32)
        self.alive[:self.initial_cracks] = 1
        self.rng_state = np.array(
            [(rng_seed + i * 2654435761) & 0xFFFFFFFFFFFFFFFF for i in range(MAX_CRACKS)],
            dtype=np.uint64)
        for ci in range(self.initial_cracks):
            self._py_find_start(ci)

    def _py_find_start(self, ci):
        found = False
        px = py = 0
        for _ in range(1000):
            px = int(self.rng.integers(0, self.W))
            py = int(self.rng.integers(0, self.H))
            if self.cgrid[py * self.W + px] < 10000:
                found = True
                break
        if found:
            a = float(self.cgrid[py * self.W + px])
            a += 90 * int(self.rng.choice([-1, 1])) + int(self.rng.integers(-2, 3))
        else:
            px = int(self.rng.integers(0, self.W))
            py = int(self.rng.integers(0, self.H))
            a  = float(self.rng.integers(0, 360))
        self.crack_x[ci] = px + 0.61 * math.cos(math.radians(a))
        self.crack_y[ci] = py + 0.61 * math.sin(math.radians(a))
        self.crack_t[ci] = a
        pi = int(self.rng.integers(0, PAL_LEN))
        self.sand_c[ci] = ALL_PALETTES[self.palette_idx, pi]
        self.sand_g[ci] = self.rng.uniform(0.01, 0.1)

    def _init_python(self):
        # start with only initial_cracks; grow to num_cracks gradually
        self.cracks = [_Crack(self.W, self.H, self.cgrid, self.rng)
                       for _ in range(self.initial_cracks)]
        self._active = self.initial_cracks

    def run(self, steps, progress_cb=None):
        self.running = True
        # vsync mode: small batches so each frame shows fine detail
        batch = 20 if self.vsync else self.BATCH

        if NUMBA:
            _run_batch(self.canvas, self.cgrid, self.W, self.H,
                       self.crack_x, self.crack_y, self.crack_t,
                       self.sand_c, self.sand_g, self.rng_state,
                       self.alive, 1, self.num_cracks,
                       self.origin_x, self.origin_y, self.origin_bias,
                       self.palette_idx)

        while self.step < steps and self.running:

            # vsync: wait until the GUI has consumed the last frame
            if self.vsync:
                self._frame_consumed.wait()
                self._frame_consumed.clear()

            with self.lock:
                if NUMBA:
                    self.alive = _run_batch(
                        self.canvas, self.cgrid, self.W, self.H,
                        self.crack_x, self.crack_y, self.crack_t,
                        self.sand_c, self.sand_g, self.rng_state,
                        self.alive, batch, self.num_cracks,
                        self.origin_x, self.origin_y, self.origin_bias,
                        self.palette_idx)

                    # use visual fill (non-white pixels) — cgrid only marks
                    # crack lines, not the sand regions between them
                    visual_filled = np.sum(np.any(self.canvas < 240, axis=2))
                    visual_ratio  = visual_filled / (self.W * self.H)

                    # ramp crack count based on visual fill
                    target = int(self.initial_cracks + (self.num_cracks - self.initial_cracks)
                                 * min(1.0, (visual_ratio / 0.8) ** 2))
                    for ci in range(self.num_cracks):
                        if int(np.sum(self.alive[:self.num_cracks])) >= target:
                            break
                        if not self.alive[ci]:
                            self.alive[ci] = 1
                            self._py_find_start(ci)

                    if (not np.any(self.alive[:self.num_cracks])
                            or visual_ratio > self.fill_pct):
                        self.step = steps
                        break
                else:
                    visual_ratio = np.sum(np.any(self.canvas < 240, axis=2)) / (self.W * self.H)
                    target = int(self.initial_cracks + (self.num_cracks - self.initial_cracks)
                                 * min(1.0, (visual_ratio / 0.8) ** 2))
                    while len(self.cracks) < target:
                        self.cracks.append(_Crack(self.W, self.H, self.cgrid, self.rng))

                    alive_count = 0
                    for crack in self.cracks:
                        if crack.alive:
                            crack.move(self.canvas)
                            alive_count += crack.alive
                    if alive_count == 0 or visual_ratio > self.fill_pct:
                        self.step = steps
                        break
                self.step += batch

            if progress_cb:
                progress_cb(min(self.step, steps), steps)

        self.running = False

    def stop(self):
        self.running = False
        self._frame_consumed.set()   # unblock vsync wait so thread can exit

    def notify_frame_consumed(self):
        """Called by GUI after each preview refresh when vsync is on."""
        self._frame_consumed.set()

    def get_image(self):
        with self.lock:
            return Image.fromarray(self.canvas.copy(), "RGB")


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Substrate Settings")
        self.resizable(False, False)
        self.result = None
        self._build()
        self.grab_set()
        self.wait_window()

    def _build(self):
        pad = dict(padx=10, pady=6)

        accel = "numba ⚡ (fast)" if NUMBA else "pure Python (pip install numba for ~35× speedup)"
        ttk.Label(self, text="Substrate Generator",
                  font=("Helvetica", 14, "bold")).pack(pady=(14, 2))
        ttk.Label(self, text=f"Engine: {accel}",
                  foreground="#080" if NUMBA else "#a00").pack(pady=(0, 8))

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=14)

        # ── Tab 1: Basic ──────────────────────────────────────────────────
        t1 = ttk.Frame(notebook, padding=10)
        notebook.add(t1, text="  Basic  ")

        SIZE_PRESETS = {
            "1080p  (1920×1080)": (1920, 1080),
            "4K     (3840×2160)": (3840, 2160),
            "Square (800×800)":   (800,  800),
            "Manual":             None,
        }
        ttk.Label(t1, text="Canvas size:").grid(row=0, column=0, sticky="e", **pad)
        self.size_preset = tk.StringVar(value="Square (800×800)")
        ttk.Combobox(t1, textvariable=self.size_preset,
                     values=list(SIZE_PRESETS.keys()),
                     state="readonly", width=20).grid(row=0, column=1, sticky="w", padx=10, pady=6)

        manual_frame = ttk.Frame(t1)
        manual_frame.grid(row=1, column=0, columnspan=2, pady=0)
        ttk.Label(manual_frame, text="Width:").pack(side="left", padx=(10,2))
        self.width_var = tk.StringVar(value="800")
        self._w_entry = ttk.Entry(manual_frame, textvariable=self.width_var, width=7)
        self._w_entry.pack(side="left")
        ttk.Label(manual_frame, text="Height:").pack(side="left", padx=(10,2))
        self.height_var = tk.StringVar(value="800")
        self._h_entry = ttk.Entry(manual_frame, textvariable=self.height_var, width=7)
        self._h_entry.pack(side="left")

        def on_preset_change(*_):
            wh = SIZE_PRESETS.get(self.size_preset.get())
            if wh:
                self.width_var.set(str(wh[0]))
                self.height_var.set(str(wh[1]))
                self._w_entry.config(state="disabled")
                self._h_entry.config(state="disabled")
            else:
                self._w_entry.config(state="normal")
                self._h_entry.config(state="normal")
        self.size_preset.trace_add("write", on_preset_change)
        on_preset_change()

        basic_fields = [
            ("Cracks",                "cracks", "100"),
            ("Steps",                 "steps",  "500000"),
            ("Seed (blank = random)", "seed",   ""),
        ]
        self.vars = {}
        for i, (label, key, default) in enumerate(basic_fields, start=2):
            ttk.Label(t1, text=label + ":").grid(row=i, column=0, sticky="e", **pad)
            v = tk.StringVar(value=default)
            ttk.Entry(t1, textvariable=v, width=14).grid(row=i, column=1, sticky="w", **pad)
            self.vars[key] = v

        base = 2 + len(basic_fields)

        ttk.Label(t1, text="Stop at fill %:").grid(row=base, column=0, sticky="e", **pad)
        fill_frame = ttk.Frame(t1)
        fill_frame.grid(row=base, column=1, sticky="w", padx=10)
        self.fill_var = tk.IntVar(value=70)
        self.fill_label = ttk.Label(fill_frame, text="70%", width=5)
        self.fill_label.pack(side="right")
        ttk.Scale(fill_frame, from_=10, to=100, orient="horizontal",
                  variable=self.fill_var, length=120,
                  command=lambda v: self.fill_label.config(
                      text=f"{int(float(v))}%")).pack(side="left")

        ttk.Label(t1, text="Output format:").grid(row=base+1, column=0, sticky="e", **pad)
        self.fmt_var = tk.StringVar(value="PNG")
        fmt_f = ttk.Frame(t1)
        fmt_f.grid(row=base+1, column=1, sticky="w", padx=10)
        ttk.Radiobutton(fmt_f, text="PNG",  variable=self.fmt_var, value="PNG").pack(side="left")
        ttk.Radiobutton(fmt_f, text="JPEG", variable=self.fmt_var, value="JPEG").pack(side="left", padx=8)

        ttk.Label(t1, text="Preview refresh (ms):").grid(row=base+2, column=0, sticky="e", **pad)
        self.refresh_var = tk.StringVar(value="150")
        ttk.Entry(t1, textvariable=self.refresh_var, width=14).grid(
            row=base+2, column=1, sticky="w", **pad)

        # ── Tab 2: Advanced ───────────────────────────────────────────────
        t2 = ttk.Frame(notebook, padding=10)
        notebook.add(t2, text="  Advanced  ")

        ttk.Label(t2, text="Controls how density builds up over time.",
                  foreground="#555").grid(row=0, column=0, columnspan=2, pady=(0,10))

        # Initial cracks slider
        ttk.Label(t2, text="Initial cracks:").grid(row=1, column=0, sticky="e", **pad)
        ic_frame = ttk.Frame(t2)
        ic_frame.grid(row=1, column=1, sticky="w", padx=10)
        self.ic_var = tk.IntVar(value=3)
        self.ic_label = ttk.Label(ic_frame, text="3", width=4)
        self.ic_label.pack(side="right")
        ttk.Scale(ic_frame, from_=1, to=20, orient="horizontal",
                  variable=self.ic_var, length=140,
                  command=lambda v: self.ic_label.config(
                      text=str(int(float(v))))).pack(side="left")
        ttk.Label(t2, text="Fewer = larger primary regions, more contrast",
                  foreground="#777").grid(row=2, column=0, columnspan=2, pady=(0,6))

        # Grid seeds slider
        ttk.Label(t2, text="Grid seeds:").grid(row=3, column=0, sticky="e", **pad)
        gs_frame = ttk.Frame(t2)
        gs_frame.grid(row=3, column=1, sticky="w", padx=10)
        self.gs_var = tk.IntVar(value=16)
        self.gs_label = ttk.Label(gs_frame, text="16", width=4)
        self.gs_label.pack(side="right")
        ttk.Scale(gs_frame, from_=1, to=64, orient="horizontal",
                  variable=self.gs_var, length=140,
                  command=lambda v: self.gs_label.config(
                      text=str(int(float(v))))).pack(side="left")
        ttk.Label(t2, text="Fewer = cracks cluster in one area",
                  foreground="#777").grid(row=4, column=0, columnspan=2, pady=(0,6))

        # Presets
        ttk.Label(t2, text="Presets:").grid(row=5, column=0, sticky="e", **pad)
        preset_f = ttk.Frame(t2)
        preset_f.grid(row=5, column=1, sticky="w", padx=10)

        def apply_preset(ic, gs):
            self.ic_var.set(ic); self.ic_label.config(text=str(ic))
            self.gs_var.set(gs); self.gs_label.config(text=str(gs))

        ttk.Button(preset_f, text="Original",
                   command=lambda: apply_preset(3, 4)).pack(side="left", padx=2)
        ttk.Button(preset_f, text="Balanced",
                   command=lambda: apply_preset(8, 16)).pack(side="left", padx=2)
        ttk.Button(preset_f, text="Uniform",
                   command=lambda: apply_preset(20, 32)).pack(side="left", padx=2)

        ttk.Label(t2,
                  text="Original: few seeds, slow ramp → dense clusters\n"
                       "Balanced: moderate contrast (default)\n"
                       "Uniform:  even coverage like a grid",
                  foreground="#555", justify="left").grid(
            row=6, column=0, columnspan=2, padx=10, pady=(6,0))

        # vsync toggle
        ttk.Separator(t2, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=10)
        self.vsync_var = tk.BooleanVar(value=False)
        vsync_cb = ttk.Checkbutton(t2, text="Slow-motion mode (vsync)",
                                   variable=self.vsync_var)
        vsync_cb.grid(row=8, column=0, columnspan=2, sticky="w", padx=10)
        ttk.Label(t2,
                  text="Locks simulation speed to the preview refresh rate.\n"
                       "Great for watching the construction step by step.",
                  foreground="#555", justify="left").grid(
            row=9, column=0, columnspan=2, padx=10, pady=(2, 0))

        # ── Tab 3: Origin ─────────────────────────────────────────────────
        t3 = ttk.Frame(notebook, padding=10)
        notebook.add(t3, text="  Origin  ")

        ttk.Label(t3, text="Controls where crack activity is concentrated.",
                  foreground="#555").grid(row=0, column=0, columnspan=2, pady=(0,10))

        # Origin position X
        ttk.Label(t3, text="Origin X:").grid(row=1, column=0, sticky="e", **pad)
        ox_frame = ttk.Frame(t3)
        ox_frame.grid(row=1, column=1, sticky="w", padx=10)
        self.ox_var = tk.DoubleVar(value=0.5)
        self.ox_label = ttk.Label(ox_frame, text="50%", width=5)
        self.ox_label.pack(side="right")
        ttk.Scale(ox_frame, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.ox_var, length=140,
                  command=lambda v: self.ox_label.config(
                      text=f"{int(float(v)*100)}%")).pack(side="left")

        # Origin position Y
        ttk.Label(t3, text="Origin Y:").grid(row=2, column=0, sticky="e", **pad)
        oy_frame = ttk.Frame(t3)
        oy_frame.grid(row=2, column=1, sticky="w", padx=10)
        self.oy_var = tk.DoubleVar(value=0.5)
        self.oy_label = ttk.Label(oy_frame, text="50%", width=5)
        self.oy_label.pack(side="right")
        ttk.Scale(oy_frame, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.oy_var, length=140,
                  command=lambda v: self.oy_label.config(
                      text=f"{int(float(v)*100)}%")).pack(side="left")

        ttk.Label(t3, text="0% = left/top edge,  100% = right/bottom edge",
                  foreground="#777").grid(row=3, column=0, columnspan=2, padx=10, pady=(0,8))

        # Origin bias strength
        ttk.Label(t3, text="Bias strength:").grid(row=4, column=0, sticky="e", **pad)
        ob_frame = ttk.Frame(t3)
        ob_frame.grid(row=4, column=1, sticky="w", padx=10)
        self.ob_var = tk.DoubleVar(value=0.0)
        self.ob_label = ttk.Label(ob_frame, text="0%", width=5)
        self.ob_label.pack(side="right")
        ttk.Scale(ob_frame, from_=0.0, to=100.0, orient="horizontal",
                  variable=self.ob_var, length=140,
                  command=lambda v: self.ob_label.config(
                      text=f"{int(float(v))}%")).pack(side="left")
        ttk.Label(t3, text="0% = uniform spread,  100% = strong pull to origin",
                  foreground="#777").grid(row=5, column=0, columnspan=2, padx=10, pady=(0,8))

        # Quick position presets
        ttk.Label(t3, text="Presets:").grid(row=6, column=0, sticky="e", **pad)
        pos_f = ttk.Frame(t3)
        pos_f.grid(row=6, column=1, sticky="w", padx=10)

        def set_origin(x, y):
            self.ox_var.set(x); self.ox_label.config(text=f"{int(x*100)}%")
            self.oy_var.set(y); self.oy_label.config(text=f"{int(y*100)}%")

        ttk.Button(pos_f, text="Top-left",
                   command=lambda: set_origin(0.05, 0.05)).pack(side="left", padx=2)
        ttk.Button(pos_f, text="Center",
                   command=lambda: set_origin(0.5, 0.5)).pack(side="left", padx=2)
        ttk.Button(pos_f, text="Bottom-right",
                   command=lambda: set_origin(0.95, 0.95)).pack(side="left", padx=2)
        ttk.Button(pos_f, text="Random",
                   command=lambda: set_origin(
                       round(random.random(), 2),
                       round(random.random(), 2))).pack(side="left", padx=2)

        # ── Tab 4: Palette ────────────────────────────────────────────────
        t4 = ttk.Frame(notebook, padding=10)
        notebook.add(t4, text="  Palette  ")

        ttk.Label(t4, text="Choose a color palette for the sand regions.",
                  foreground="#555").grid(row=0, column=0, columnspan=2, pady=(0,10))

        self.palette_var = tk.StringVar(value="Vivid")

        for i, name in enumerate(PALETTE_NAMES):
            row = i + 1
            rb = ttk.Radiobutton(t4, text=name, variable=self.palette_var, value=name)
            rb.grid(row=row, column=0, sticky="w", padx=10, pady=3)

            # Draw a small color swatch
            swatch = tk.Canvas(t4, width=180, height=16, highlightthickness=0)
            swatch.grid(row=row, column=1, padx=10, pady=3, sticky="w")
            pal = PALETTES[name]
            sw = 180 // len(pal)
            for j, color in enumerate(pal):
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                swatch.create_rectangle(j*sw, 0, (j+1)*sw, 16,
                                        fill=f'#{r:02x}{g:02x}{b:02x}',
                                        outline='')

        # ── Buttons ───────────────────────────────────────────────────────
        btn = ttk.Frame(self)
        btn.pack(pady=12)
        ttk.Button(btn, text="Generate", command=self._ok).pack(side="left", padx=6)
        ttk.Button(btn, text="Cancel",   command=self.destroy).pack(side="left", padx=6)

    def _ok(self):
        try:
            w       = int(self.width_var.get())
            h       = int(self.height_var.get())
            c       = int(self.vars["cracks"].get())
            s       = int(self.vars["steps"].get())
            seed_s  = self.vars["seed"].get().strip()
            seed    = int(seed_s) if seed_s else None
            refresh = int(self.refresh_var.get())
            assert all(v > 0 for v in [w, h, c, s, refresh])
        except Exception:
            messagebox.showerror("Invalid input", "Please check your values.", parent=self)
            return
        self.result = dict(width=w, height=h, cracks=c, steps=s,
                           seed=seed, refresh=refresh,
                           fill_pct=int(self.fill_var.get()),
                           fmt=self.fmt_var.get(),
                           initial_cracks=int(self.ic_var.get()),
                           num_seeds=int(self.gs_var.get()),
                           vsync=self.vsync_var.get(),
                           origin_x=round(self.ox_var.get(), 3),
                           origin_y=round(self.oy_var.get(), 3),
                           origin_bias=round(self.ob_var.get() / 100.0 * 0.04, 5),
                           palette_idx=PALETTE_NAMES.index(self.palette_var.get()))
        self.destroy()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SubstrateApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Substrate")
        self.configure(bg="#1e1e1e")
        self.engine    = None
        self._after_id = None
        self.params    = None
        self._build_ui()
        self.after(100, self._show_settings)

    def _build_ui(self):
        toolbar = tk.Frame(self, bg="#2d2d2d", pady=6)
        toolbar.pack(fill="x")

        # ttk style so macOS actually respects the colors
        style = ttk.Style()
        style.configure("Toolbar.TButton",
                         font=("Helvetica", 12),
                         padding=(10, 5))

        ttk.Button(toolbar, text="⚙  Settings", style="Toolbar.TButton",
                   command=self._show_settings).pack(side="left", padx=6, pady=2)
        ttk.Button(toolbar, text="⏹  Stop",     style="Toolbar.TButton",
                   command=self._stop).pack(side="left", padx=4, pady=2)
        ttk.Button(toolbar, text="💾  Save PNG", style="Toolbar.TButton",
                   command=self._save).pack(side="left", padx=4, pady=2)

        badge_bg  = "#1a6b2a" if NUMBA else "#7a3a00"
        badge_txt = "⚡ numba" if NUMBA else "🐢 python"
        tk.Label(toolbar, text=badge_txt,
                 bg=badge_bg, fg="white",
                 font=("Helvetica", 11, "bold"),
                 padx=10, pady=4).pack(side="right", padx=8)

        self.status_var = tk.StringVar(value="Configure settings to begin.")
        tk.Label(self, textvariable=self.status_var,
                 bg="#1e1e1e", fg="#aaa", anchor="w", padx=8).pack(fill="x", side="bottom")
        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", side="bottom")

        self.tk_canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.tk_canvas.pack(fill="both", expand=True)
        self._img_id = None
        self._tk_img = None

    def _show_settings(self):
        self._stop()
        dlg = SettingsDialog(self)
        if dlg.result:
            self.params = dlg.result
            self._start()

    def _start(self):
        p = self.params
        self.status_var.set("Compiling JIT…" if NUMBA else "Starting…")
        self.update_idletasks()

        # Scale crack count and seeds by linear canvas dimension
        # (sqrt of area ratio) so density looks consistent across sizes.
        # initial_cracks is a structural choice — not scaled.
        BASE = 800.0
        linear_scale = math.sqrt(p["width"] * p["height"]) / BASE

        scaled_cracks  = max(1, int(round(p["cracks"]    * linear_scale)))
        scaled_steps   = max(1, int(round(p["steps"]     * linear_scale)))
        scaled_seeds   = max(1, int(round(p["num_seeds"] * linear_scale)))
        initial_cracks = p["initial_cracks"]   # intentionally not scaled

        self.engine = SubstrateEngine(p["width"], p["height"],
                                      scaled_cracks,
                                      p["seed"], p["fill_pct"],
                                      initial_cracks, scaled_seeds,
                                      p["vsync"],
                                      p["origin_x"], p["origin_y"],
                                      p["origin_bias"],
                                      p["palette_idx"])
        self.progress["value"] = 0

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(p['width']+20, sw-40)}x{min(p['height']+80, sh-80)}")
        self.update_idletasks()   # ensure canvas has real dimensions before first refresh

        self.status_var.set(
            f"Running  {p['width']}×{p['height']}  "
            f"cracks={scaled_cracks} (ic={initial_cracks})  stop={p['fill_pct']}% fill"
            + ("  🎞 vsync" if p["vsync"] else "")
            + ("  ⚡" if NUMBA else ""))

        threading.Thread(
            target=self.engine.run,
            args=(p["steps"], self._progress_cb),
            daemon=True).start()

        self._schedule_refresh(p["refresh"])

    def _progress_cb(self, step, total):
        self.progress["value"] = int(100 * step / total)

    def _schedule_refresh(self, ms):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(ms, lambda: self._refresh(ms))

    def _refresh(self, ms):
        if self.engine is None:
            return
        pil_img = self.engine.get_image()

        # fit image into the canvas widget — use winfo after update so size is real
        self.tk_canvas.update_idletasks()
        cw = self.tk_canvas.winfo_width()
        ch = self.tk_canvas.winfo_height()

        if cw > 4 and ch > 4:
            # scale down only if image is larger than the widget
            iw, ih = pil_img.size
            scale = min(cw / iw, ch / ih, 1.0)
            if scale < 1.0:
                nw, nh = int(iw * scale), int(ih * scale)
                pil_img = pil_img.resize((nw, nh), Image.NEAREST)

        self._tk_img = ImageTk.PhotoImage(pil_img)
        cx, cy = cw // 2, ch // 2

        if self._img_id is None:
            self._img_id = self.tk_canvas.create_image(
                cx, cy, anchor="center", image=self._tk_img)
        else:
            self.tk_canvas.coords(self._img_id, cx, cy)
            self.tk_canvas.itemconfig(self._img_id, image=self._tk_img)

        if self.engine.running:
            self._after_id = self.after(ms, lambda: self._refresh(ms))
        else:
            self.status_var.set("Done. Use 💾 Save PNG to export.")
            self.progress["value"] = 100

        # signal vsync — let simulation thread render the next batch
        if self.engine:
            self.engine.notify_frame_consumed()

    def _stop(self):
        if self.engine:
            self.engine.stop()
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _save(self):
        if self.engine is None:
            messagebox.showinfo("Nothing to save", "Run a simulation first.")
            return
        fmt = self.params.get("fmt", "PNG") if self.params else "PNG"
        ext = ".jpg" if fmt == "JPEG" else ".png"
        default_name = f"substrate{ext}"
        filetypes = [("JPEG image", "*.jpg")] if fmt == "JPEG" else [("PNG image", "*.png")]
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=filetypes + [("All files", "*.*")],
            initialfile=default_name)
        if path:
            img = self.engine.get_image()
            save_kw = {"quality": 95, "optimize": True} if fmt == "JPEG" else {}
            img.save(path, format=fmt, **save_kw)
            self.status_var.set(f"Saved → {path}")

    def on_close(self):
        self._stop()
        self.destroy()


if __name__ == "__main__":
    app = SubstrateApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

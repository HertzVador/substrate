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
# Palette  — MUST be defined before @njit functions so numba captures it
# ---------------------------------------------------------------------------

PALETTE_NB = np.array([
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
], dtype=np.float64)

PAL_LEN = len(PALETTE_NB)
MAX_CRACKS = 512

# ---------------------------------------------------------------------------
# Numba JIT inner loop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _run_batch(canvas, cgrid, W, H,
               crack_x, crack_y, crack_t,
               sand_c, sand_g,
               rng_state,
               alive,
               batch_size, num_cracks):
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

            # walk perpendicular to find region boundary
            rx = crack_x[ci]
            ry = crack_y[ci]
            for _w in range(2000):
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

            # restart or retire — inlined find_start
            if needs_restart:
                found = False
                fpx = 0
                fpy = 0
                for _t in range(200):   # 200 tries is enough; 1000 was too slow on full grids
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    fpx = int(v * W)
                    if fpx >= W: fpx = W - 1
                    s = (s * M1 + M2) & MASK
                    v = float(s >> np.uint64(33)) / 2147483648.0
                    fpy = int(v * H)
                    if fpy >= H: fpy = H - 1
                    if cgrid[fpy * W + fpx] < 10000:
                        found = True
                        break

                if not found:
                    alive[ci] = 0   # grid is full — retire this crack
                else:
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
                    sand_c[ci, 0] = PALETTE_NB[pi, 0]
                    sand_c[ci, 1] = PALETTE_NB[pi, 1]
                    sand_c[ci, 2] = PALETTE_NB[pi, 2]
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
                 initial_cracks=3, num_seeds=16):
        self.W = width
        self.H = height
        self.num_cracks    = min(num_cracks, MAX_CRACKS)
        self.initial_cracks = max(1, min(initial_cracks, num_cracks))
        self.step = 0
        self.running = False
        self.lock = threading.Lock()
        self.fill_pct = max(1, min(100, fill_pct)) / 100.0

        rng_seed = seed if seed is not None else random.randint(0, 2**31)
        self.rng = np.random.default_rng(rng_seed)

        self.canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        self.cgrid  = np.full(width * height, 10001, dtype=np.int32)

        # seed initial crack directions — spread across canvas
        n_seeds = max(1, min(num_seeds, 256))
        for _ in range(n_seeds):
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
        self.sand_c[ci] = PALETTE_NB[pi]
        self.sand_g[ci] = self.rng.uniform(0.01, 0.1)

    def _init_python(self):
        # start with only initial_cracks; grow to num_cracks gradually
        self.cracks = [_Crack(self.W, self.H, self.cgrid, self.rng)
                       for _ in range(self.initial_cracks)]
        self._active = self.initial_cracks

    def run(self, steps, progress_cb=None):
        self.running = True
        batch = self.BATCH

        if NUMBA:
            _run_batch(self.canvas, self.cgrid, self.W, self.H,
                       self.crack_x, self.crack_y, self.crack_t,
                       self.sand_c, self.sand_g, self.rng_state,
                       self.alive, 1, self.num_cracks)

        while self.step < steps and self.running:
            with self.lock:
                if NUMBA:
                    # gradually activate more cracks as the grid fills
                    active = int(np.sum(self.alive[:self.num_cracks]))
                    filled_ratio = np.sum(self.cgrid < 10000) / (self.W * self.H)
                    # ramp: at 0% fill → initial_cracks, at 50% fill → num_cracks
                    target = int(self.initial_cracks + (self.num_cracks - self.initial_cracks)
                                 * min(1.0, filled_ratio / 0.5))
                    for ci in range(self.num_cracks):
                        if active >= target:
                            break
                        if not self.alive[ci]:
                            self.alive[ci] = 1
                            self._py_find_start(ci)
                            active += 1

                    self.alive = _run_batch(
                        self.canvas, self.cgrid, self.W, self.H,
                        self.crack_x, self.crack_y, self.crack_t,
                        self.sand_c, self.sand_g, self.rng_state,
                        self.alive, batch, self.num_cracks)

                    filled = np.sum(self.cgrid < 10000)
                    if (not np.any(self.alive[:self.num_cracks])
                            or filled > self.W * self.H * self.fill_pct):
                        self.step = steps
                        break
                else:
                    # gradually add cracks for pure-python path too
                    filled_ratio = np.sum(self.cgrid < 10000) / (self.W * self.H)
                    target = int(self.initial_cracks + (self.num_cracks - self.initial_cracks)
                                 * min(1.0, filled_ratio / 0.5))
                    while len(self.cracks) < target:
                        self.cracks.append(_Crack(self.W, self.H, self.cgrid, self.rng))

                    alive_count = 0
                    for crack in self.cracks:
                        if crack.alive:
                            crack.move(self.canvas)
                            alive_count += crack.alive
                    if alive_count == 0:
                        self.step = steps
                        break
                self.step += batch

            if progress_cb:
                progress_cb(min(self.step, steps), steps)

        self.running = False

    def stop(self):
        self.running = False

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
        self.fill_var = tk.IntVar(value=30)
        self.fill_label = ttk.Label(fill_frame, text="30%", width=5)
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
                           num_seeds=int(self.gs_var.get()))
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

        self.engine = SubstrateEngine(p["width"], p["height"], p["cracks"],
                                      p["seed"], p["fill_pct"],
                                      p["initial_cracks"], p["num_seeds"])
        self.progress["value"] = 0

        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(p['width']+20, sw-40)}x{min(p['height']+80, sh-80)}")
        self.update_idletasks()   # ensure canvas has real dimensions before first refresh

        self.status_var.set(
            f"Running  {p['width']}×{p['height']}  "
            f"cracks={p['cracks']}  steps={p['steps']:,}  "
            f"stop={p['fill_pct']}% fill"
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

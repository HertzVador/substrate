# Substrate

A Python generative art tool inspired by [Jared Tarbell's *Substrate* (2004)](http://www.complexification.net/gallery/machines/substrate/) from complexification.net.

Cracks grow across a canvas, branching perpendicularly off one another and filling enclosed regions with soft watercolor-like sand. The result is an organic, tile-like structure that emerges from a handful of simple rules.

---

## Features

- **Live preview** — watch the image build in real time as the simulation runs
- **Numba JIT acceleration** — ~35× faster than pure Python on the inner loop; falls back gracefully if numba isn't installed
- **Configurable settings** — canvas size, crack count, step limit, random seed, preview refresh rate, and fill % stop condition
- **Stop at fill %** — slider to control how densely the canvas fills before the simulation ends (10–100%)
- **Save PNG** — export the full-resolution image at any point, including mid-run
- **Reproducible** — set a seed to get the same image every time

---

## Requirements

- Python 3.8+
- [Pillow](https://pillow.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [numba](https://numba.readthedocs.io/) *(optional but strongly recommended)*

Runs on macOS, Windows, and Linux. `tkinter` is included with standard Python on macOS and Windows. On Linux you may need to install it separately:

```bash
sudo apt install python3-tk   # Debian / Ubuntu
sudo dnf install python3-tkinter  # Fedora
```

---

## Quick start

```bash
./run.sh
```

`run.sh` installs any missing dependencies automatically, then launches the GUI.

Or manually:

```bash
pip install Pillow numpy numba
python substrate_gui.py
```

---

## GUI

When the app opens, a settings dialog appears:

| Setting         | Description                                                 |
| -----------------| -------------------------------------------------------------|
| Width / Height  | Canvas size in pixels                                       |
| Cracks          | Number of simultaneous crack lines                          |
| Steps           | Maximum simulation iterations                               |
| Seed            | Fixed seed for reproducible output (leave blank for random) |
| Stop at fill %  | Halt when this fraction of the grid is covered              |
| Preview refresh | How often the live preview updates (milliseconds)           |

The toolbar shows the active engine (`⚡ numba` or `🐢 python`) and provides **Stop**, and **Save PNG** buttons.

---

## Command-line (headless)

`substrate.py` can generate images without a GUI:

```bash
python substrate.py                          # 800×800, default settings
python substrate.py 1920 1080               # custom size
python substrate.py 1920 1080 --cracks 150 --steps 800000 --seed 42 --output art.png
```

| Argument | Default | Description |
|---|---|---|
| `width height` | 800 800 | Canvas dimensions |
| `--cracks` | 100 | Simultaneous crack lines |
| `--steps` | 500000 | Simulation iterations |
| `--seed` | random | Random seed |
| `--output` | substrate.png | Output file path |

---

## How it works

The algorithm maintains a grid where each cell stores the angle of the crack that passed through it. On each step:

1. Every active crack advances one pixel in its current direction with a slight wobble.
2. It walks perpendicular to itself until it hits another crack or the canvas edge, defining a region boundary.
3. A sand painter deposits semi-transparent colored grains from the crack position to that boundary, building up a watercolor wash.
4. If a crack hits another crack at a significant angle, it dies and a new one spawns perpendicular to an existing crack elsewhere.
5. The simulation ends when all cracks can no longer find empty space to start from, or when the configured fill % is reached.

The numba path compiles the entire inner loop (crack movement, region walk, sand painting, grid update) to native code via `@njit`. The first run takes ~2 seconds to compile; subsequent runs use a disk cache and start instantly.

---

## Credits

Inspired by **Jared Tarbell**, *Substrate* (2004)  
[complexification.net/gallery/machines/substrate](http://www.complexification.net/gallery/machines/substrate/)

Original algorithm and Processing source released as open source by Jared Tarbell.

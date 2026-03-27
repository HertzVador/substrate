"""
Substrate generative art - Python implementation
Based on Jared Tarbell's Substrate Watercolor (2004)
complexification.net

Usage:
    python substrate.py                    # 800x800
    python substrate.py 1920 1080
    python substrate.py 1920 1080 --cracks 100 --steps 500000 --output art.png
"""

import argparse
import math
import random
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Color palette  (warm earthy tones similar to Tarbell's pollockShimmering)
# ---------------------------------------------------------------------------

PALETTE = [
    (214, 178, 120), (190, 140,  80), (160, 100,  50), (220, 200, 160),
    (180, 160, 130), (200, 180, 140), (240, 220, 180), (160, 130,  90),
    (100,  70,  40), (230, 210, 170), (170, 140, 100), (210, 190, 150),
    (140, 110,  70), (250, 230, 190), (120,  90,  55), (195, 165, 115),
    (175, 145,  95), (205, 175, 125), (185, 155, 105), (165, 135,  85),
    ( 80,  55,  30), (245, 225, 185), (155, 125,  75), (225, 205, 165),
    (135, 105,  65), (215, 195, 155), (145, 115,  70), (235, 215, 175),
    (125,  95,  60), (255, 235, 195), (115,  85,  50), (205, 185, 145),
    # some cooler tones for variety
    (120, 140, 160), ( 90, 110, 140), (150, 170, 190), ( 70,  90, 120),
    (100, 130, 150), (140, 160, 180), ( 60,  80, 110), (130, 150, 170),
    (160, 130, 110), (180, 150, 130), (200, 170, 150), (140, 110,  90),
]


def some_color():
    return random.choice(PALETTE)


def color_with_alpha(base_color, alpha, bg=(255, 255, 255)):
    """Blend base_color onto bg with given alpha (0-1)."""
    r = int(bg[0] * (1 - alpha) + base_color[0] * alpha)
    g = int(bg[1] * (1 - alpha) + base_color[1] * alpha)
    b = int(bg[2] * (1 - alpha) + base_color[2] * alpha)
    return (r, g, b)


# ---------------------------------------------------------------------------
# Sand Painter
# ---------------------------------------------------------------------------

class SandPainter:
    def __init__(self):
        self.c = some_color()
        self.g = random.uniform(0.01, 0.1)

    def render(self, img_pixels, width, height, x, y, ox, oy):
        """Paint sand grains from (ox,oy) toward (x,y)."""
        # Modulate gain
        self.g += random.uniform(-0.050, 0.050)
        self.g = max(0.0, min(1.0, self.g))

        grains = 64
        w = self.g / (grains - 1)

        for i in range(grains):
            # Sinusoidal distribution — denser at edges
            t_val = math.sin(math.sin(i * w))
            px = int(ox + (x - ox) * t_val)
            py = int(oy + (y - oy) * t_val)
            if 0 <= px < width and 0 <= py < height:
                alpha = 0.1 - i / (grains * 10.0)
                alpha = max(0.0, alpha)
                existing = img_pixels[px, py]
                blended = color_with_alpha(self.c, alpha, existing)
                img_pixels[px, py] = blended


# ---------------------------------------------------------------------------
# Crack
# ---------------------------------------------------------------------------

class Crack:
    def __init__(self, width, height, cgrid):
        self.width = width
        self.height = height
        self.cgrid = cgrid
        self.x = 0.0
        self.y = 0.0
        self.t = 0.0
        self.sp = SandPainter()
        self._find_start()

    def _find_start(self):
        found = False
        timeout = 0
        px, py = 0, 0
        while not found and timeout < 1000:
            timeout += 1
            px = random.randint(0, self.width - 1)
            py = random.randint(0, self.height - 1)
            if self.cgrid[py * self.width + px] < 10000:
                found = True

        if found:
            a = self.cgrid[py * self.width + px]
            a += 90 * random.choice([-1, 1]) + random.randint(-2, 2)
            self._start_crack(px, py, a)
        else:
            # fallback
            self._start_crack(
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
                random.randint(0, 360)
            )

    def _start_crack(self, x, y, t):
        self.x = float(x)
        self.y = float(y)
        self.t = float(t)
        self.x += 0.61 * math.cos(math.radians(t))
        self.y += 0.61 * math.sin(math.radians(t))

    def move(self, img_pixels, img):
        # Advance
        self.x += 0.42 * math.cos(math.radians(self.t))
        self.y += 0.42 * math.sin(math.radians(self.t))

        fuzz = 0.33
        cx = int(self.x + random.uniform(-fuzz, fuzz))
        cy = int(self.y + random.uniform(-fuzz, fuzz))

        # Draw sand region color
        self._region_color(img_pixels, img)

        # Draw semi-transparent crack pixel
        if 0 <= int(self.x) < self.width and 0 <= int(self.y) < self.height:
            existing = img_pixels[int(self.x), int(self.y)]
            crack_color = color_with_alpha((0, 0, 0), 85 / 255.0, existing)
            img_pixels[int(self.x), int(self.y)] = crack_color

        # Bounds check
        if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
            self._find_start()
            return True

        idx = cy * self.width + cx
        existing_angle = self.cgrid[idx]

        if existing_angle > 10000 or abs(existing_angle - self.t) < 5:
            # Continue cracking
            self.cgrid[idx] = int(self.t)
        elif abs(existing_angle - self.t) > 2:
            # Hit another crack — restart
            self._find_start()

        return True

    def _region_color(self, img_pixels, img):
        """Walk perpendicular to crack until hitting a boundary, then paint sand."""
        rx, ry = self.x, self.y
        open_space = True

        while open_space:
            rx += 0.81 * math.sin(math.radians(self.t))
            ry -= 0.81 * math.cos(math.radians(self.t))
            cx = int(rx)
            cy = int(ry)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                if self.cgrid[cy * self.width + cx] > 10000:
                    pass  # open space, keep walking
                else:
                    open_space = False
            else:
                open_space = False

        self.sp.render(img_pixels, self.width, self.height, rx, ry, self.x, self.y)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_substrate(width=800, height=800, num_cracks=100,
                       steps=500000, seed=None, output="substrate.png"):
    if seed is not None:
        random.seed(seed)

    print(f"Generating {width}x{height} Substrate image...")

    img = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = img.load()

    EMPTY = 10001
    cgrid = [EMPTY] * (width * height)

    # Seed a few initial crack directions
    for _ in range(16):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        cgrid[y * width + x] = random.randint(0, 360)

    # Start initial cracks
    cracks = [Crack(width, height, cgrid) for _ in range(min(3, num_cracks))]

    for step in range(steps):
        for crack in cracks:
            crack.move(pixels, img)

        # Gradually grow to num_cracks
        if len(cracks) < num_cracks and step % 200 == 0:
            cracks.append(Crack(width, height, cgrid))

        if step % (steps // 10) == 0 and step > 0:
            pct = int(100 * step / steps)
            print(f"  {pct}% complete...")

    img.save(output)
    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate Substrate generative art")
    parser.add_argument("width",  nargs="?", type=int, default=800)
    parser.add_argument("height", nargs="?", type=int, default=800)
    parser.add_argument("--cracks", type=int, default=100,
                        help="Max simultaneous cracks (default: 100)")
    parser.add_argument("--steps", type=int, default=500000,
                        help="Simulation steps (default: 500000)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="substrate.png")
    args = parser.parse_args()

    generate_substrate(
        width=args.width,
        height=args.height,
        num_cracks=args.cracks,
        steps=args.steps,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()

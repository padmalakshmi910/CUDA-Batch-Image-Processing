#!/usr/bin/env python3
"""Generate deterministic RGB PPM images using Python's standard library."""
import math, random
from pathlib import Path

input_dir = Path("input")
input_dir.mkdir(exist_ok=True)
count, width, height = 200, 256, 256

for image_id in range(count):
    rng = random.Random(image_id)
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            wave = int(127.5 + 127.5 * math.sin((x + image_id * 7) * 0.035)
                      * math.cos((y + image_id * 3) * 0.027))
            pixels.extend(((wave + rng.randrange(32)) % 256,
                           (x + image_id * 11) % 256,
                           (y * 2 + image_id * 5) % 256))
    with (input_dir / f"image_{image_id:04d}.ppm").open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        f.write(pixels)

print(f"Generated {count} images of size {width}x{height}.")

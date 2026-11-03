"""
Bulk positive-image downloader for the 1,477-site geoglyph catalog.

Builds on the previous team's tile-stitching approach (mt0.google.com/vt),
but fixes several issues found in v2gmapDownloader.py:

  1. Zoom is now computed PER SITE from its actual size (a width / b width),
     instead of a fixed zoom=20 for every site. Small (~30m) and large
     (~200m) geoglyphs both get a sensibly framed image.
  2. Tile grid stays fixed at 8x8 (matches V2segmentationMaskGenerator.py's
     TILES_W/TILES_H=8 assumption) -- only the zoom (ground coverage) varies.
  3. HTTP status codes are checked; failed/blank tiles are logged, not
     silently stitched in.
  4. Requests are throttled and retried with backoff -- 1,477 sites x 64
     tiles/site is ~94k requests; hammering the endpoint risks IP blocking.
  5. Resumable: already-downloaded codes are skipped on rerun.
  6. Writes a manifest CSV (code, lat, lon, zoom, window_m, status) for
     your methods section / reproducibility, and for the verification
     step that comes next (mask overlay).
  7. Grayscale conversion is dropped by default (pass --grayscale to keep
     the old behavior) -- there's no clear reason to discard color
     channels before training a classifier.

NOTE ON PROVENANCE: this still uses the unofficial mt0.google.com/vt
endpoint (no API key, not a licensed source). That's a real decision to
revisit for a published paper -- consider the official Google Maps
Platform Static API (or another licensed/citable imagery source) if
reviewers or reproducibility matter to you, which given your stated goal
of a public benchmark, they probably do.
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

TILE_SIZE = 256
TILE_COUNT = 8  # fixed to match V2segmentationMaskGenerator's 8x8 assumption
MIN_ZOOM = 14
MAX_ZOOM = 21  # imagery tends to degrade/blank out beyond this in most areas
MARGIN_FACTOR = 3.0  # window should be ~3x the shape's longest dimension
MIN_WINDOW_M = 150.0  # floor, so tiny shapes still get some context


class GoogleMapsLayers:
    SATELLITE = "s"


def meters_per_pixel(lat_deg, zoom):
    return 156543.03392 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def compute_zoom(lat_deg, shape_dim_m):
    """Pick the integer zoom whose 8x8-tile window best fits the target size."""
    target_width_m = max(shape_dim_m * MARGIN_FACTOR, MIN_WINDOW_M)
    window_px = TILE_SIZE * TILE_COUNT
    best_zoom, best_diff = MAX_ZOOM, float("inf")
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        window_m = meters_per_pixel(lat_deg, z) * window_px
        diff = abs(window_m - target_width_m)
        if diff < best_diff:
            best_diff, best_zoom = diff, z
    return best_zoom


def get_xy(lat, lng, zoom):
    num_tiles = 1 << zoom
    point_x = int((TILE_SIZE / 2 + lng * TILE_SIZE / 360.0) * num_tiles // TILE_SIZE)
    sin_y = math.sin(lat * math.pi / 180.0)
    point_y = int(
        (TILE_SIZE / 2 + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(TILE_SIZE / (2 * math.pi)))
        * num_tiles
        // TILE_SIZE
    )
    return point_x, point_y


def fetch_tile(session, x, y, zoom, layer, max_retries=4, timeout=10):
    url = f"https://mt0.google.com/vt?lyrs={layer}&x={x}&y={y}&z={zoom}"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200 and resp.content:
                return resp.content
        except requests.RequestException:
            pass
        time.sleep(0.5 * (2 ** attempt))  # exponential backoff
    return None


def generate_image(session, lat, lng, zoom, layer=GoogleMapsLayers.SATELLITE):
    start_x, start_y = get_xy(lat, lng, zoom)
    width, height = TILE_SIZE * TILE_COUNT, TILE_SIZE * TILE_COUNT
    map_img = Image.new("RGB", (width, height))
    missing_tiles = 0

    for dx in range(-TILE_COUNT // 2, TILE_COUNT // 2):
        for dy in range(-TILE_COUNT // 2, TILE_COUNT // 2):
            content = fetch_tile(session, start_x + dx, start_y + dy, zoom, layer)
            if content is None:
                missing_tiles += 1
                continue
            tmp_path = f"_tile_{os.getpid()}_{dx}_{dy}.tmp"
            with open(tmp_path, "wb") as f:
                f.write(content)
            try:
                tile_img = Image.open(tmp_path)
                map_img.paste(tile_img, ((dx + TILE_COUNT // 2) * TILE_SIZE, (dy + TILE_COUNT // 2) * TILE_SIZE))
            finally:
                os.remove(tmp_path)
            time.sleep(0.05)  # gentle throttle between tile requests

    return map_img, missing_tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="amazon_geoglyphs.xlsx")
    ap.add_argument("--out_dir", default="positives")
    ap.add_argument("--manifest", default="download_manifest.csv")
    ap.add_argument("--sleep_between_sites", type=float, default=0.5)
    ap.add_argument("--forms", nargs="*", default=None, help="Only these form values (default: all)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.xlsx, engine="openpyxl")
    if args.forms:
        df = df[df["form"].isin(args.forms)]

    # Resume support: load already-processed codes from an existing manifest
    done_codes = set()
    manifest_path = Path(args.manifest)
    write_header = not manifest_path.exists()
    if manifest_path.exists():
        with manifest_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done_codes.add(row["code"])

    session = requests.Session()

    with manifest_path.open("a", newline="") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(["code", "form", "lat", "lon", "zoom", "window_m", "missing_tiles", "status", "out_path"])

        for _, row in df.iterrows():
            code = str(row["code"]).strip()
            if code in done_codes:
                continue

            lat, lon = row["lat"], row["lon"]
            form = row.get("form", "unknown")
            a_w, b_w = row.get("a width"), row.get("b width")
            dims = [d for d in (a_w, b_w) if isinstance(d, (int, float))]
            shape_dim = max(dims) if dims else MIN_WINDOW_M

            zoom = compute_zoom(lat, shape_dim)
            window_m = meters_per_pixel(lat, zoom) * TILE_SIZE * TILE_COUNT

            out_path = Path(args.out_dir) / f"geoglyph_{form}_#{code}.png"
            print(f"[{code}] form={form} dim={shape_dim:.0f}m -> zoom={zoom} window={window_m:.0f}m")

            try:
                img, missing = generate_image(session, lat, lon, zoom)
                status = "ok" if missing < 8 else "too_many_missing_tiles"
                if status == "ok":
                    img.save(out_path)
            except Exception as e:
                print(f"  FAILED: {e}")
                missing, status, out_path = -1, "error", ""

            writer.writerow([code, form, lat, lon, zoom, f"{window_m:.0f}", missing, status, out_path])
            mf.flush()
            time.sleep(args.sleep_between_sites)


if __name__ == "__main__":
    main()

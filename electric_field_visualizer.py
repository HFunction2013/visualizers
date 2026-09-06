#!/usr/bin/env python3
"""
Point Charge Electric Field Line Visualizer
=============================================
Controls:
  - Left-click on empty space: create a new charge (default electron charge -e)
  - Left-click on a charge: select and drag
  - Right-click on a charge: set charge value (input bar at top, Enter to confirm / ESC to cancel)
  - Delete / Backspace: delete the selected charge
  - Mouse wheel: zoom around the cursor
  - Middle-mouse drag: pan the view
  - R: reset view   C: clear all charges   ESC: deselect

Performance:
  - Field lines are cached in world coordinates; recomputed only when charges change
    (panning and zooming do NOT invalidate the cache).
  - Adaptive backend: pure Python per-line tracing for small sets (< NUMPY_THRESHOLD lines),
    NumPy-vectorized batch tracing for larger sets.
  - Thread-pool parallelism is enabled automatically on multi-core machines with very large
    line counts (>= 4 cores and > THREADPOOL_THRESHOLD lines).
"""

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import pygame
except ImportError:
    print("Please install pygame first:  pip install pygame")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ──────────────────────────── Constants ────────────────────────────
WIDTH, HEIGHT = 1280, 800
BG_COLOR = (18, 18, 28)
GRID_COLOR = (38, 38, 54)
GRID_MAJOR_COLOR = (55, 55, 78)
POSITIVE_COLOR = (255, 70, 70)
NEGATIVE_COLOR = (70, 130, 255)
SELECTED_RING = (255, 230, 80)
FIELD_LINE_COLOR = (170, 170, 195)
ARROW_COLOR = (200, 200, 225)
TEXT_COLOR = (220, 220, 235)
HUD_COLOR = (140, 140, 165)
INPUT_TEXT = (255, 255, 200)

K = 1.0                        # Coulomb constant (relative units)
E_CHARGE = 1.0                 # elementary charge magnitude (relative units)
CHARGE_SCREEN_RADIUS = 13      # charge display radius on screen (pixels)
FIELD_LINE_STEP = 0.012         # field-line integration step (world coords)
MAX_FIELD_LINE_POINTS = 3000    # max points per field line
MIN_LINES_PER_CHARGE = 8        # minimum field lines per charge
LINES_PER_UNIT_CHARGE = 12      # field lines per unit charge
ARROW_SPACING_PX = 90           # arrow spacing along field lines (pixels)

# World-coordinate radii for tracing (zoom-independent, so cache survives pan/zoom)
START_RADIUS = 0.04
TERMINATE_RADIUS = 0.06

# Adaptive backend thresholds
NUMPY_THRESHOLD = 160           # use NumPy when total lines >= this
CPU_COUNT = os.cpu_count() or 2
THREADPOOL_THRESHOLD = 400      # use thread pool when lines >= this AND cores >= 4
MAX_WORKERS = min(CPU_COUNT, 8)


# ──────────────────────────── Charge class ────────────────────────────
class Charge:
    __slots__ = ("x", "y", "q")

    def __init__(self, x, y, q):
        self.x = x  # world coordinate
        self.y = y
        self.q = q  # charge in units of e

    @property
    def is_positive(self):
        return self.q > 0

    @property
    def color(self):
        return POSITIVE_COLOR if self.is_positive else NEGATIVE_COLOR


# ──────────────────────────── Camera class ────────────────────────────
class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 55.0

    def world_to_screen(self, wx, wy, sw, sh):
        return ((wx - self.x) * self.zoom + sw / 2.0,
                (wy - self.y) * self.zoom + sh / 2.0)

    def screen_to_world(self, sx, sy, sw, sh):
        return ((sx - sw / 2.0) / self.zoom + self.x,
                (sy - sh / 2.0) / self.zoom + self.y)

    def zoom_at(self, sx, sy, factor, sw, sh):
        wx0, wy0 = self.screen_to_world(sx, sy, sw, sh)
        self.zoom = max(5.0, min(8000.0, self.zoom * factor))
        wx1, wy1 = self.screen_to_world(sx, sy, sw, sh)
        self.x += wx0 - wx1
        self.y += wy0 - wy1


# ──────────────────────────── Field-line tracing ────────────────────────────
def _build_starts(charges):
    """Build (starts, directions) for all field lines."""
    starts = []
    directions = []
    for c in charges:
        n = max(MIN_LINES_PER_CHARGE, int(round(abs(c.q) * LINES_PER_UNIT_CHARGE)))
        d = 1 if c.q > 0 else -1
        for i in range(n):
            a = 2.0 * math.pi * i / n
            starts.append((c.x + START_RADIUS * math.cos(a),
                           c.y + START_RADIUS * math.sin(a)))
            directions.append(d)
    return starts, directions


def _trace_single_python(charges, start, direction):
    """Pure-Python tracing for a single field line. Returns list of (x, y) tuples."""
    points = [start]
    x, y = start
    for _ in range(MAX_FIELD_LINE_POINTS):
        Ex, Ey = 0.0, 0.0
        for c in charges:
            dx, dy = x - c.x, y - c.y
            r2 = dx * dx + dy * dy
            if r2 < 1e-12:
                continue
            r = math.sqrt(r2)
            E = K * c.q / r2
            Ex += E * dx / r
            Ey += E * dy / r
        E_mag = math.hypot(Ex, Ey)
        if E_mag < 1e-14:
            break
        inv = FIELD_LINE_STEP / E_mag
        x += direction * Ex * inv
        y += direction * Ey * inv

        # Opposite-charge termination
        for c in charges:
            if direction == 1 and c.q < 0 and math.hypot(x - c.x, y - c.y) < TERMINATE_RADIUS:
                points.append((c.x, c.y))
                return points
            if direction == -1 and c.q > 0 and math.hypot(x - c.x, y - c.y) < TERMINATE_RADIUS:
                points.append((c.x, c.y))
                return points
        points.append((x, y))
    return points


def _trace_numpy(charge_pos, charge_q, starts, directions, max_steps=MAX_FIELD_LINE_POINTS):
    """
    NumPy-vectorized batch tracing. Only still-active lines are processed each step,
    and hit-detection runs every step for exact termination.
    charge_pos: (M, 2) array, charge_q: (M,) array.
    Returns list of lists of (x, y) tuples in world coordinates.
    """
    n = len(starts)
    if n == 0:
        return []

    pos = np.array(starts, dtype=np.float64)      # (N, 2)
    dirs = np.array(directions, dtype=np.float64)  # (N,)
    active = np.ones(n, dtype=bool)

    traj = np.zeros((n, max_steps + 1, 2), dtype=np.float64)
    traj[:, 0, :] = pos
    counts = np.ones(n, dtype=np.int32)

    neg_mask = charge_q < 0   # (M,)
    pos_mask = charge_q > 0
    cp0 = charge_pos[:, 0]
    cp1 = charge_pos[:, 1]

    for _ in range(max_steps):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        p = pos[idx]       # (A, 2)
        dr = dirs[idx]     # (A,)

        # Electric field via broadcasting: (A, M)
        dx = p[:, 0:1] - cp0[None, :]
        dy = p[:, 1:2] - cp1[None, :]
        r2 = dx * dx + dy * dy
        np.maximum(r2, 1e-12, out=r2)
        r = np.sqrt(r2)
        E = K * charge_q[None, :] / r2
        Ex = np.sum(E * dx / r, axis=1)   # (A,)
        Ey = np.sum(E * dy / r, axis=1)
        E_mag = np.sqrt(Ex * Ex + Ey * Ey)

        # Zero-field termination
        zero = E_mag < 1e-14
        if zero.any():
            active[idx[zero]] = False
            keep = ~zero
            if not keep.any():
                break
            p = p[keep]
            dr = dr[keep]
            Ex = Ex[keep]
            Ey = Ey[keep]
            E_mag = E_mag[keep]
            idx = idx[keep]

        # Advance positions
        safe = np.maximum(E_mag, 1e-14)
        p[:, 0] += dr * Ex / safe * FIELD_LINE_STEP
        p[:, 1] += dr * Ey / safe * FIELD_LINE_STEP
        pos[idx] = p

        # Opposite-charge hit detection (every step for precision)
        dist = np.sqrt((p[:, 0:1] - cp0[None, :]) ** 2 +
                       (p[:, 1:2] - cp1[None, :]) ** 2)
        hit = (((dr[:, None] == 1) & neg_mask[None, :]) |
               ((dr[:, None] == -1) & pos_mask[None, :])) & (dist < TERMINATE_RADIUS)
        hit_line = hit.any(axis=1)
        if hit_line.any():
            hit_ci = hit[hit_line].argmax(axis=1)
            hit_global = idx[hit_line]
            traj[hit_global, counts[hit_global], :] = charge_pos[hit_ci]
            counts[hit_global] += 1
            active[hit_global] = False

        # Record still-active positions
        still = idx[active[idx]]
        traj[still, counts[still], :] = pos[still]
        counts[still] += 1

    return [traj[i, :counts[i], :].tolist() for i in range(n)]


def compute_field_lines(charges):
    """
    Compute all field lines in world coordinates using the adaptive backend.
    Returns (lines, backend_label).
    """
    if not charges:
        return [], "idle"

    starts, directions = _build_starts(charges)
    n = len(starts)

    # Small sets: pure Python (better early-termination, lower overhead)
    if not HAS_NUMPY or n < NUMPY_THRESHOLD:
        lines = [_trace_single_python(charges, s, d) for s, d in zip(starts, directions)]
        return lines, "Python"

    # Larger sets: NumPy vectorized
    charge_pos = np.array([[c.x, c.y] for c in charges], dtype=np.float64)
    charge_q = np.array([c.q for c in charges], dtype=np.float64)

    # Thread pool only on >=4 cores with very large line counts (measured to be slower otherwise)
    if CPU_COUNT >= 4 and n >= THREADPOOL_THRESHOLD:
        workers = min(MAX_WORKERS, max(2, n // 200))
        batch = (n + workers - 1) // workers
        batches = [(charge_pos, charge_q, starts[i:i + batch], directions[i:i + batch])
                   for i in range(0, n, batch)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_trace_numpy, *args) for args in batches]
            lines = []
            for f in futures:
                lines.extend(f.result())
        return lines, f"NumPy+{workers}threads"

    lines = _trace_numpy(charge_pos, charge_q, starts, directions)
    return lines, "NumPy"


# ──────────────────────────── Drawing helpers ────────────────────────────
def draw_arrow(surface, x, y, angle, color, size=7):
    p1 = (x + size * math.cos(angle), y + size * math.sin(angle))
    p2 = (x + size * math.cos(angle + 2.5), y + size * math.sin(angle + 2.5))
    p3 = (x + size * math.cos(angle - 2.5), y + size * math.sin(angle - 2.5))
    pygame.draw.polygon(surface, color, [p1, p2, p3])


def draw_grid(surface, sw, sh, camera):
    target_px = 90.0
    world_spacing = target_px / camera.zoom
    mag = 10 ** math.floor(math.log10(world_spacing))
    norm = world_spacing / mag
    if norm < 1.5:
        world_spacing = mag
    elif norm < 3.5:
        world_spacing = 2.0 * mag
    else:
        world_spacing = 5.0 * mag

    left = camera.x - sw / 2.0 / camera.zoom
    right = camera.x + sw / 2.0 / camera.zoom
    top = camera.y - sh / 2.0 / camera.zoom
    bottom = camera.y + sh / 2.0 / camera.zoom

    x = math.floor(left / world_spacing) * world_spacing
    while x <= right:
        sx, _ = camera.world_to_screen(x, 0, sw, sh)
        is_major = abs(x / world_spacing) % 5 < 0.001
        color = GRID_MAJOR_COLOR if is_major else GRID_COLOR
        pygame.draw.line(surface, color, (sx, 0), (sx, sh), 1)
        x += world_spacing

    y = math.floor(top / world_spacing) * world_spacing
    while y <= bottom:
        _, sy = camera.world_to_screen(0, y, sw, sh)
        is_major = abs(y / world_spacing) % 5 < 0.001
        color = GRID_MAJOR_COLOR if is_major else GRID_COLOR
        pygame.draw.line(surface, color, (0, sy), (sw, sy), 1)
        y += world_spacing


def draw_field_lines(surface, world_lines, camera, sw, sh):
    """Project cached world-coordinate lines to screen and draw."""
    cx_off = -camera.x * camera.zoom + sw / 2.0
    cy_off = -camera.y * camera.zoom + sh / 2.0
    z = camera.zoom

    for line in world_lines:
        if len(line) < 2:
            continue
        pts = [(p[0] * z + cx_off, p[1] * z + cy_off) for p in line]
        pygame.draw.lines(surface, FIELD_LINE_COLOR, False, pts, 1)

        acc = 0.0
        for j in range(1, len(pts)):
            seg = math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
            acc += seg
            if acc >= ARROW_SPACING_PX:
                dx = pts[j][0] - pts[j - 1][0]
                dy = pts[j][1] - pts[j - 1][1]
                if dx * dx + dy * dy > 1e-6:
                    draw_arrow(surface, pts[j][0], pts[j][1],
                               math.atan2(dy, dx), ARROW_COLOR, size=6)
                acc = 0.0


def draw_charges(surface, charges, camera, sw, sh, selected, font, small_font):
    for c in charges:
        sx, sy = camera.world_to_screen(c.x, c.y, sw, sh)
        r = CHARGE_SCREEN_RADIUS

        if c is selected:
            pygame.draw.circle(surface, SELECTED_RING, (int(sx), int(sy)), r + 5, 2)

        pygame.draw.circle(surface, c.color, (int(sx), int(sy)), r)
        pygame.draw.circle(surface, (255, 255, 255), (int(sx), int(sy)), r, 1)
        pygame.draw.circle(surface, (255, 255, 255, 40),
                           (int(sx - r * 0.3), int(sy - r * 0.3)), r // 3)

        sign = "+" if c.is_positive else "\u2212"
        t = font.render(sign, True, (255, 255, 255))
        surface.blit(t, (sx - t.get_width() // 2, sy - t.get_height() // 2))

        label = f"{c.q:g}e"
        lt = small_font.render(label, True, TEXT_COLOR)
        surface.blit(lt, (sx + r + 5, sy - lt.get_height() // 2))


# ──────────────────────────── Interaction helpers ────────────────────────────
def find_charge_at(charges, wx, wy, camera):
    threshold = max(CHARGE_SCREEN_RADIUS / camera.zoom * 1.5, TERMINATE_RADIUS)
    best = None
    best_d = threshold
    for c in charges:
        d = math.hypot(wx - c.x, wy - c.y)
        if d < best_d:
            best_d = d
            best = c
    return best


# ──────────────────────────── Main loop ────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Point Charge Electric Field Visualizer")
    clock = pygame.time.Clock()

    camera = Camera()
    charges = [
        Charge(-2.0, 0.0, 1.0),
        Charge(2.0, 0.0, -1.0),
    ]
    selected = None

    # Field-line cache in world coordinates + dirty flag
    field_lines_cache = []
    field_lines_dirty = True
    backend_label = "idle"

    # Interaction state
    dragging = False
    drag_charge = None
    drag_off_x = 0.0
    drag_off_y = 0.0
    panning = False
    pan_last = (0, 0)
    typing = False
    typing_charge = None
    input_text = ""

    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 20)
    hud_font = pygame.font.SysFont(None, 17)

    running = True
    while running:
        sw, sh = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                wx, wy = camera.screen_to_world(mx, my, sw, sh)

                if typing:
                    continue

                if event.button == 1:  # left
                    hit = find_charge_at(charges, wx, wy, camera)
                    if hit:
                        selected = hit
                        dragging = True
                        drag_charge = hit
                        drag_off_x = hit.x - wx
                        drag_off_y = hit.y - wy
                    else:
                        nc = Charge(wx, wy, -E_CHARGE)
                        charges.append(nc)
                        selected = nc
                        field_lines_dirty = True

                elif event.button == 2:  # middle
                    panning = True
                    pan_last = (mx, my)

                elif event.button == 3:  # right
                    hit = find_charge_at(charges, wx, wy, camera)
                    if hit:
                        selected = hit
                        typing = True
                        typing_charge = hit
                        input_text = f"{hit.q:g}"

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    drag_charge = None
                elif event.button == 2:
                    panning = False

            elif event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                if dragging and drag_charge:
                    wx, wy = camera.screen_to_world(mx, my, sw, sh)
                    drag_charge.x = wx + drag_off_x
                    drag_charge.y = wy + drag_off_y
                    field_lines_dirty = True
                elif panning:
                    dx = mx - pan_last[0]
                    dy = my - pan_last[1]
                    camera.x -= dx / camera.zoom
                    camera.y -= dy / camera.zoom
                    pan_last = (mx, my)
                    # NOTE: panning does NOT invalidate the cache (world-coordinate cache)

            elif event.type == pygame.MOUSEWHEEL:
                if typing:
                    continue
                mx, my = pygame.mouse.get_pos()
                factor = 1.12 if event.y > 0 else 1.0 / 1.12
                camera.zoom_at(mx, my, factor, sw, sh)
                # NOTE: zooming does NOT invalidate the cache

            elif event.type == pygame.KEYDOWN:
                if typing:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        try:
                            val = float(input_text)
                            if abs(val) > 1e-9:
                                typing_charge.q = val
                                field_lines_dirty = True
                        except ValueError:
                            pass
                        typing = False
                        typing_charge = None
                        input_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        typing = False
                        typing_charge = None
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        ch = event.unicode
                        if ch and ch in "0123456789.-" and len(input_text) < 24:
                            input_text += ch
                else:
                    if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        if selected:
                            charges.remove(selected)
                            selected = None
                            field_lines_dirty = True
                    elif event.key == pygame.K_ESCAPE:
                        selected = None
                    elif event.key == pygame.K_c:
                        charges.clear()
                        selected = None
                        field_lines_dirty = True
                    elif event.key == pygame.K_r:
                        camera.x, camera.y, camera.zoom = 0.0, 0.0, 55.0

        # ────────────── Render ──────────────
        # Recompute only when charge configuration changed; pan/zoom reuse cache
        if field_lines_dirty:
            field_lines_cache, backend_label = compute_field_lines(charges)
            field_lines_dirty = False

        screen.fill(BG_COLOR)
        draw_grid(screen, sw, sh, camera)
        draw_field_lines(screen, field_lines_cache, camera, sw, sh)
        draw_charges(screen, charges, camera, sw, sh, selected, font, small_font)

        # HUD
        hud = [
            "Left-click empty: new (-e)  |  Left-click charge: select/drag  |  Right-click charge: set charge",
            "Delete: remove selected  |  Wheel: zoom  |  Middle-drag: pan  |  R: reset view  |  C: clear all",
        ]
        for i, line in enumerate(hud):
            t = hud_font.render(line, True, HUD_COLOR)
            screen.blit(t, (10, sh - 36 + i * 16))

        if selected and not typing:
            info = f"Selected  q = {selected.q:g} e    pos = ({selected.x:.3f}, {selected.y:.3f})"
            t = small_font.render(info, True, SELECTED_RING)
            screen.blit(t, (12, 10))

        numpy_status = "on" if HAS_NUMPY else "off"
        cnt = small_font.render(
            f"Charges: {len(charges)}  |  Lines: {len(field_lines_cache)}  |  "
            f"Zoom: {camera.zoom:.0f}  |  Backend: {backend_label}  |  NumPy: {numpy_status}  |  CPU: {CPU_COUNT}",
            True, HUD_COLOR)
        screen.blit(cnt, (sw - cnt.get_width() - 12, 10))

        # Input bar
        if typing:
            bar = pygame.Surface((sw, 56), pygame.SRCALPHA)
            bar.fill((0, 0, 0, 190))
            screen.blit(bar, (0, 0))
            pygame.draw.line(screen, INPUT_TEXT, (0, 56), (sw, 56), 1)

            prompt = f"Set charge (units of e, Enter to confirm / ESC to cancel):  {input_text}"
            t = font.render(prompt, True, INPUT_TEXT)
            screen.blit(t, (20, 14))

            if (pygame.time.get_ticks() // 530) % 2 == 0:
                cx = 20 + t.get_width() + 2
                pygame.draw.line(screen, INPUT_TEXT, (cx, 14), (cx, 42), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()

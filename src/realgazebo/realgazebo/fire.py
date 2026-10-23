"""Fire spread for the manager's MSG 7/8/9 fires.

Pure Python (no ROS / gz) so it unit-tests like protocol.py. A fire is a
star-shaped polygon: RAY_COUNT rays from the ignition point. Only the wind
moves it: each ray grows with the wind component blowing along it and burns
taller there, so the fire line stretches downwind. With no wind the fire
keeps its ignition ring. Times are simulation seconds.
"""
import math

# Fire line vertices per fire: 5 + 2 + 16 * 32 = 519 B per state packet,
# well inside protocol.MAX_FIRE_STATE_BYTES.
RAY_COUNT = 32
MAX_FIRE_ID = 255   # fire_id is a u8 on the wire
MIN_RADIUS = 0.5    # m, a zero-radius ignite still starts as a small ring
WIND_SPREAD = 0.15  # m/s of spread per m/s of wind blowing along the ray
BASE_FLAME = 1.0    # m
WIND_FLAME = 0.5    # extra m of flame per m/s of wind along the ray

_DIRS = [(math.cos(2 * math.pi * i / RAY_COUNT),
          math.sin(2 * math.pi * i / RAY_COUNT)) for i in range(RAY_COUNT)]


class _Fire:
    def __init__(self, position, radius, now):
        self.center = position
        self.radii = [max(radius, MIN_RADIUS)] * RAY_COUNT
        self.flames = [BASE_FLAME] * RAY_COUNT
        self.last_t = now


class FireModel:
    def __init__(self):
        self._fires = {}

    def ids(self):
        return sorted(self._fires)

    def free_id(self):
        """Smallest fire_id not burning, or None when all 256 are taken."""
        return next(
            (i for i in range(MAX_FIRE_ID + 1) if i not in self._fires), None)

    def ignite(self, fire_id, position, radius, now) -> bool:
        """Start a fire. False (nothing changes) if fire_id is already alive."""
        if fire_id in self._fires:
            return False
        self._fires[fire_id] = _Fire(position, radius, now)
        return True

    def extinguish(self, fire_id) -> bool:
        """Remove a fire. False if it was not alive."""
        return self._fires.pop(fire_id, None) is not None

    def step(self, now, wind_xy):
        """Advance every fire to `now` under the horizontal wind (m/s).

        Time going backwards (sim reset) moves nothing.
        """
        wx, wy = wind_xy
        for fire in self._fires.values():
            dt = max(0.0, now - fire.last_t)
            fire.last_t = now
            for i, (dx, dy) in enumerate(_DIRS):
                along = max(0.0, wx * dx + wy * dy)
                fire.radii[i] += WIND_SPREAD * along * dt
                fire.flames[i] = BASE_FLAME + WIND_FLAME * along

    def polygons(self, fire_id):
        """The fire line as one part: [[(x, y, z, flame_len), ...]]."""
        fire = self._fires[fire_id]
        cx, cy, cz = fire.center
        return [[(cx + r * dx, cy + r * dy, cz, flame)
                 for (dx, dy), r, flame in zip(_DIRS, fire.radii, fire.flames)]]

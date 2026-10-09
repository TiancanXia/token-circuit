"""Fast-forward itinerary and timing. Emits stage indices; never computes tensors."""

from collections import deque
from dataclasses import dataclass
import math
from .maps import all_groups, doorway, houses, road_edges, approach_door


@dataclass(frozen=True)
class Leg:
    house: str
    block: int = -1
    indices: tuple = ()

    @property
    def point(self):
        return doorway(self.house)


def road_path(start, end):
    """Shortest door-to-door route in the same graph used by walking/rendering."""
    neighbors = {h.key: [] for h in houses()}
    for a, b in road_edges():
        neighbors[a.key].append(b.key)
        neighbors[b.key].append(a.key)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        key, path = queue.popleft()
        if key == end:
            return path
        for neighbor in neighbors[key]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
    raise ValueError(f"No road from {start} to {end}")


class Flight:
    speed = 1200.0
    step_seconds = 0.055
    trail_seconds = 0.24

    def __init__(self, stages, next_index, position):
        self.x, self.y = position
        nearest = approach_door(position)
        self.legs = [Leg(nearest)]
        current = nearest
        for group in all_groups(stages):
            indices = tuple(i for i in group.indices if i >= next_index)
            if not indices:
                continue
            path = road_path(current, group.house)
            self.legs.extend(Leg(key, group.block) for key in path[:-1])
            self.legs.append(Leg(group.house, group.block, indices))
            current = group.house
        self.leg_index = 0
        self.stage_cursor = 0
        self.dwell = 0.0
        self.clock = 0.0
        self.history = deque([(self.x, self.y, 0.0)])
        self.visited = []

    @property
    def finished(self):
        return self.leg_index >= len(self.legs)

    @property
    def active_leg(self):
        return self.legs[min(self.leg_index, len(self.legs) - 1)]

    @property
    def trail(self):
        return tuple(
            (x, y, max(0.0, 1 - (self.clock - at) / self.trail_seconds))
            for x, y, at in self.history
        )

    def advance(self, dt):
        """Consume time, including travel and per-stage dwell, exactly once."""
        self.clock += dt
        while self.history and self.clock - self.history[0][2] > self.trail_seconds:
            self.history.popleft()
        events = []
        budget = dt
        while budget > 1e-9 and not self.finished:
            leg = self.active_leg
            tx, ty = leg.point
            distance = math.hypot(tx - self.x, ty - self.y)
            if distance > 1e-8:
                travel = min(budget, distance / self.speed)
                fraction = min(1.0, travel * self.speed / distance)
                self.x += (tx - self.x) * fraction
                self.y += (ty - self.y) * fraction
                budget -= travel
                self.history.append((self.x, self.y, self.clock - budget))
                if fraction < 1.0:
                    break
            if self.stage_cursor < len(leg.indices):
                elapsed = min(budget, self.step_seconds - self.dwell)
                self.dwell += elapsed
                budget -= elapsed
                if self.dwell + 1e-9 < self.step_seconds:
                    break
                events.append(leg.indices[self.stage_cursor])
                self.stage_cursor += 1
                self.dwell = 0.0
                continue
            if leg.indices:
                self.visited.append((leg.house, leg.block))
            self.leg_index += 1
            self.stage_cursor = 0
        return tuple(events)

"""Scene transitions, walkable maps and autoregressive rounds; no UI dependency."""

from dataclasses import replace
import math
from ..model.forward import ForwardPass
from ..model.cache import KVCache
from .rooms import Rooms, ROOMS
from .attention_probe import AttentionProbe
from .maps import (
    houses,
    all_groups,
    group_for,
    checkpoints,
    EXIT,
    macro_floor,
    micro_floor,
    doorway,
)
from .flight import Flight
from .transfer import Transfer
from ..model.prompt import demo_config, join_tokens, quote_tokens
from ..i18n import Locale


class Session:
    speed = 155.0
    window_size = 64
    birth_seconds = Transfer.duration

    def __init__(
        self, config=None, seed=17, guided=False, language="zh", cache_enabled=False
    ):
        self.language = language
        self.locale = Locale(language)
        self.initial_config = config or demo_config(language=language)
        self.guided = guided
        tokens = tuple(
            self.initial_config.vocab[i] for i in self.initial_config.context
        )
        self.prompt_text = join_tokens(tokens, language)
        self.quote_demo = tokens == quote_tokens(language)
        self.seed = seed
        self.generated = []
        self.generation = 0
        self.scene = "macro"
        self.group = None
        self.cache_enabled = cache_enabled
        self.cache = KVCache()
        self.rooms = Rooms()
        self.attention_probe = AttentionProbe()
        context = (
            self.initial_config.context[:1]
            if guided
            else self.initial_config.context[-self.window_size :]
        )
        self.forward = self.new_forward(context)
        self.x, self.y = 110.0, 245.0
        self.macro_position = (self.x, self.y)
        self.time = self.walk_time = 0.0
        self.moving = False
        self.facing = (0, 1)
        self.glows = {h.key: 0.0 for h in houses()}
        self.message = "E → Embedding"
        self.message_until = 5.0
        self.birth_time = 0.0
        self.pending_token = None
        self.flash = 0.0
        self.flight = None
        self.transfer = None

    def new_forward(self, context):
        return ForwardPass(
            replace(self.initial_config, context=context),
            self.seed + self.generation,
            self.cache if self.cache_enabled else None,
        )

    @property
    def room_checkpoints(self):
        return (
            self.rooms.active.spec.checkpoints()
            if self.scene == "detail"
            else checkpoints(self.group)
        )

    def checkpoint_complete(self, index):
        return index < (
            self.rooms.active.progress
            if self.scene == "detail"
            else self.forward.next_index
        )

    def checkpoint_title(self, index):
        if self.scene == "detail":
            spec = self.rooms.active.spec
            return (
                self.tr(spec.label_keys[index])
                if spec.label_keys
                else spec.nodes[index]
            )
        stage = self.forward.stages[index]
        return "Next token" if self.guided and stage.kind == "sample" else stage.title

    def cycle_head(self):
        if self.scene != "detail":
            return False
        visit = self.rooms.active
        visit.head = (visit.head + 1) % self.forward.config.heads
        self.say(self.tr("head_hint", n=visit.head + 1))
        return True

    @property
    def token(self):
        return self.forward.config.vocab[self.forward.config.context[-1]]

    @property
    def sentence_cells(self):
        if not self.guided:
            return tuple(
                self.forward.config.vocab[i] for i in self.forward.config.context
            )
        return tuple(
            self.initial_config.vocab[i] if pos <= self.generation else None
            for pos, i in enumerate(self.initial_config.context)
        )

    @property
    def expected_group(self):
        if self.forward.next_index >= len(self.forward.stages):
            return None
        return group_for(self.forward.stages, self.forward.next_index)

    @property
    def block(self):
        if self.scene == "flight" and self.flight.active_leg.block >= 0:
            return self.flight.active_leg.block
        expected = self.expected_group
        return (
            expected.block
            if expected and expected.block >= 0
            else len(self.forward.config.blocks) - 1
            if self.forward.next_index > 0
            else 0
        )

    @property
    def near_house(self):
        if self.scene != "macro":
            return None
        h = min(houses(), key=lambda h: (h.x - self.x) ** 2 + (h.y + 30 - self.y) ** 2)
        return h if abs(h.x - self.x) <= 72 and abs(h.y + 30 - self.y) <= 64 else None

    @property
    def next_checkpoint(self):
        if self.group is None:
            return None
        index = (
            self.rooms.active.progress
            if self.scene == "detail"
            else self.forward.next_index
        )
        return next((c for c in self.room_checkpoints if c.index == index), None)

    @property
    def near_checkpoint(self):
        if self.scene not in ("micro", "detail"):
            return None
        cp = min(
            self.room_checkpoints,
            key=lambda c: (c.x - self.x) ** 2 + (c.y - self.y) ** 2,
        )
        return cp if abs(cp.x - self.x) < 69 and abs(cp.y - self.y) < 51 else None

    @property
    def group_complete(self):
        if self.scene == "detail":
            return self.rooms.active.complete
        return (
            self.group is not None and self.group.indices[-1] < self.forward.next_index
        )

    def say(self, message):
        self.message, self.message_until = message, self.time + 4.0

    def tr(self, key, **values):
        return self.locale.text(key, **values)

    def house_group(self, key):
        candidates = [g for g in all_groups(self.forward.stages) if g.house == key]
        candidate = next(
            (g for g in candidates if g.block == self.block),
            candidates[-1] if candidates else None,
        )
        if candidate and candidate.indices[0] <= self.forward.next_index:
            return candidate
        completed = [g for g in candidates if g.indices[-1] < self.forward.next_index]
        return completed[-1] if completed else None

    def interact(self):
        if self.scene == "flight":
            return False
        if self.scene == "done":
            self.say(self.tr("done_restart"))
            return False
        if self.scene == "birth":
            self.say(self.tr("returning"))
            return False
        if self.scene == "macro":
            house = self.near_house
            if not house:
                self.say(self.tr("near_enter"))
                return False
            group = self.house_group(house.key)
            if not group:
                expected = self.expected_group
                title = next(h.title for h in houses() if h.key == expected.house)
                self.say(self.tr("next_house", title=title))
                return False
            self.group = group
            self.macro_position = (self.x, self.y)
            self.scene = "micro"
            cp = self.next_checkpoint or checkpoints(group)[0]
            self.x, self.y = cp.x, cp.y + 22
            self.moving = False
            self.say(self.tr("room_hint"))
            return True
        if math.hypot(self.x - EXIT[0], self.y - EXIT[1]) < 58:
            if self.group_complete:
                return self.return_parent()
            self.say(self.tr("finish_first"))
            return False
        cp = self.near_checkpoint
        if not cp:
            self.say(self.tr("near_process"))
            return False
        if self.scene == "detail":
            return self.rooms.interact(self, cp)
        if (
            self.forward.stages[cp.index].kind in ROOMS
            and cp.index <= self.forward.next_index
        ):
            return self.rooms.enter(self, cp.index)
        if cp.index != self.forward.next_index:
            if cp.index < self.forward.next_index:
                self.say(self.tr("already_done"))
            else:
                self.say(
                    self.tr(
                        "process_first",
                        title=self.forward.stages[self.forward.next_index].title,
                    )
                )
            return False
        self._execute_step(cp.index)
        return True

    def _execute_step(self, index):
        """Both manual and automatic paths share the same ordered executor."""
        result = self.forward.execute(index)
        self.flash = 1.0
        self.say(
            self.tr(
                "step_done",
                title=result.stage.title,
                next=self.tr("leave_room" if self.group_complete else "next_node"),
            )
        )
        if result.stage.kind == "sample":
            if self.guided and self.generation == len(self.initial_config.context) - 1:
                self.scene = "done"
                self.x, self.y = doorway("output")
                self.moving = False
                self.say(self.tr("done_restart"))
                return True
            self.pending_token = (
                self.initial_config.vocab[
                    self.initial_config.context[self.generation + 1]
                ]
                if self.guided
                else self.forward.sampled
            )
            self.scene = "birth"
            self.x, self.y = doorway("output")
            self.transfer = Transfer(
                (self.x, self.y), doorway("embedding"), self.seed + self.generation
            )
            self.birth_time = 0.0
            self.moving = False
            self.say(self.tr("revealing", token=self.pending_token))

    def start_fast_forward(self):
        if (
            self.scene not in ("macro", "micro", "detail")
            or self.expected_group is None
        ):
            return False
        if self.scene == "detail":
            self.return_parent()
        if self.scene == "micro":
            self.return_macro()
        self.flight = Flight(
            self.forward.stages, self.forward.next_index, (self.x, self.y)
        )
        self.transfer = None
        self.scene, self.group = "flight", None
        self.moving = True
        self.say(self.tr("fast_message"))
        return True

    def return_parent(self):
        if self.scene == "detail":
            return self.rooms.back(self)
        return self.return_macro()

    def return_macro(self):
        if self.scene != "micro":
            self.say(self.tr("done_restart" if self.scene == "done" else "enter"))
            return False
        self.scene = "macro"
        self.x, self.y = self.macro_position
        self.moving = False
        expected = self.expected_group
        if expected:
            title = next(h.title for h in houses() if h.key == expected.house)
            layer = (
                " / " + self.tr("layer", n=expected.block + 1)
                if expected.block >= 0
                else ""
            )
            self.say(self.tr("next_house", title=title + layer))
        return True

    def cycle_room_layer(self):
        if self.scene == "detail":
            self.return_parent()
        if self.scene != "micro":
            self.say(self.tr("tab_inside"))
            return False
        candidates = [
            g
            for g in all_groups(self.forward.stages)
            if g.house == self.group.house and g.indices[0] <= self.forward.next_index
        ]
        if len(candidates) < 2:
            self.say(self.tr("only_layer"))
            return False
        self.group = candidates[(candidates.index(self.group) + 1) % len(candidates)]
        cp = self.next_checkpoint or checkpoints(self.group)[0]
        self.x, self.y = cp.x, cp.y + 22
        self.moving = False
        self.say(self.tr("layer_message", n=self.group.block + 1))
        return True

    def jump_to(self, index):
        """Recompute the current prefix up to (not including) a chosen stage."""
        if not isinstance(index, int) or not 0 <= index < len(self.forward.stages):
            raise ValueError("Target must be a valid stage index")
        forward = self.new_forward(self.forward.config.context)
        for previous in range(index):
            forward.execute(previous)
        self.forward = forward
        self.rooms = Rooms()
        self.attention_probe = AttentionProbe()
        self.flight = self.transfer = self.pending_token = None
        self.birth_time = self.flash = 0.0
        self.scene = "micro"
        self.group = group_for(forward.stages, index)
        cp = next(cp for cp in checkpoints(self.group) if cp.index == index)
        self.x, self.y = cp.x, cp.y + 22
        self.macro_position = doorway(self.group.house)
        self.moving = False
        self.glows = {h.key: float(h.key == self.group.house) for h in houses()}
        title = forward.stages[index].title
        if self.group.block >= 0:
            title = self.tr("layer", n=self.group.block + 1) + " / " + title
        self.say(self.tr("jumped", title=title))
        return True

    def start_next_round(self):
        if self.scene != "birth" or self.pending_token is None:
            return False
        token = self.pending_token
        idx = self.forward.config.vocab.index(token)
        self.generated.append(token)
        self.generation += 1
        context = (self.forward.config.context + (idx,))[-self.window_size :]
        self.forward = self.new_forward(context)
        self.rooms = Rooms()
        self.attention_probe = AttentionProbe()
        self.scene, self.group = "macro", None
        self.x, self.y = doorway("embedding")
        self.pending_token = None
        self.birth_time = 0.0
        self.glows = {h.key: 0.0 for h in houses()}
        self.say(self.tr("new_token", token=token))
        return True

    def can_walk(self, x, y):
        floor = (
            macro_floor
            if self.scene in ("macro", "flight", "birth", "done")
            else micro_floor
        )
        return all(
            floor(x + dx, y + dy)
            for dx, dy in ((0, 0), (3, 0), (-3, 0), (0, 3), (0, -3))
        )

    def update(self, dt, dx=0, dy=0):
        dt = max(0.0, min(dt, 0.1))
        self.time += dt
        self.flash = max(0.0, self.flash - dt * 1.7)
        if self.scene == "birth":
            self.birth_time += dt
            self.transfer.elapsed = min(self.birth_seconds, self.birth_time)
            if self.birth_time >= self.birth_seconds:
                self.start_next_round()
            return
        if self.scene == "done":
            self.moving = False
            return
        if self.scene == "flight":
            old = (self.x, self.y)
            events = self.flight.advance(dt)
            self.x, self.y = self.flight.x, self.flight.y
            self.moving = (self.x, self.y) != old
            if self.moving:
                self.facing = (self.x - old[0], self.y - old[1])
                self.walk_time += dt
            for index in events:
                self._execute_step(index)
            active = self.flight.active_leg.house
            for key, value in self.glows.items():
                target = 1.0 if key == active else 0.0
                self.glows[key] = value + (target - value) * min(1.0, dt * 12)
            return
        self.moving = bool(dx or dy)
        if self.moving:
            distance = math.hypot(dx, dy)
            dx, dy = dx / distance, dy / distance
            self.facing = (dx, dy)
            self.walk_time += dt
            steps = max(1, math.ceil(self.speed * dt / 2))
            sx, sy = dx * self.speed * dt / steps, dy * self.speed * dt / steps
            for _ in range(steps):
                if self.can_walk(self.x + sx, self.y):
                    self.x += sx
                if self.can_walk(self.x, self.y + sy):
                    self.y += sy
        near = self.near_house
        for key, value in self.glows.items():
            target = 1.0 if near and near.key == key else 0.0
            self.glows[key] = value + (target - value) * min(1.0, dt * 4)

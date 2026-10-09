"""Resolve the selected stage consistently for the HUD and model guide."""

from ..simulation.maps import checkpoints
from .charts import result_for


def focused_stage(s):
    if s.scene == "detail":
        visit = s.rooms.active
        return s.forward.stages[visit.index], visit.result
    if s.scene in ("birth", "done"):
        return s.forward.stages[-1], s.forward.results[-1]
    if s.scene == "flight":
        index = max(0, s.forward.next_index - 1)
        return s.forward.stages[index], result_for(s, index)
    if s.scene == "micro":
        cp = s.near_checkpoint or s.next_checkpoint or checkpoints(s.group)[-1]
        return s.forward.stages[cp.index], result_for(s, cp.index)
    house = s.near_house
    group = s.house_group(house.key) if house else s.expected_group
    if group:
        index = group.indices[0]
    else:
        index = s.forward.next_index
    return s.forward.stages[min(index, len(s.forward.stages) - 1)], result_for(s, index)

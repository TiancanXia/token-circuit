"""Developer-only driver for newly nested rooms; mirrors player checkpoint order."""
from tokenwalk.simulation.maps import EXIT


def complete_child(game, on_node=None):
    s=game.session
    if s.scene!='detail':
        return
    for cp in s.room_checkpoints:
        if cp.index<s.rooms.active.progress:
            continue
        s.x,s.y=cp.x,cp.y
        game.interact()
        if on_node:
            on_node(game)
    s.x,s.y=EXIT
    game.interact()
    assert s.scene=='micro'

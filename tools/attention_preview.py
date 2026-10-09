"""Optional Canvas reconstructions of genuine cached decoding, not screenshots."""
import tkinter as tk
from dataclasses import replace
from tokenwalk.app import Game
from tokenwalk.simulation.maps import group_for, checkpoints
from .preview import capture, verify_panel


def advance_to_prefix(game, generation=5):
    s=game.session
    while s.generation<generation:
        s.start_fast_forward()
        current=s.generation
        for _ in range(1000):
            s.update(1/30)
            if s.generation>current:
                break
        else:
            raise AssertionError('Forward did not complete')
    s.forward.execute(0)
    s.forward.execute(1)
    s.group=group_for(s.forward.stages,2)
    s.scene='micro'
    s.macro_position=(500,245)
    return s


def main():
    root=tk.Tk()
    root.withdraw()
    game=Game(root,animate=False)
    try:
        for language in ('zh','en'):
            if game.language!=language:
                game.toggle_language()
            if game.config.num_kv_heads==game.config.heads:
                game.toggle_attention()
            game.canvas.winfo_width=lambda:1260
            game.canvas.winfo_height=lambda:854
            s=advance_to_prefix(game)
            assert s.forward.cache_reused==5
            for cp in checkpoints(s.group):
                s.x,s.y=cp.x,cp.y
                game.interact()
                kind=s.rooms.active.spec.kind
                for node in s.room_checkpoints:
                    s.x,s.y=node.x,node.y
                    game.interact()
                    assert s.rooms.active.progress==node.index+1
                    verify_panel(game)
                    if node.index in {'qkv':(1,4),'rope':(1,2),'attention':(0,1,2,3,4,5),'project':(0,1)}[kind]:
                        capture(game,f'preview-detail-{kind}-{node.index}-{language}.png')
                    if kind=='attention' and node.index==0 and language=='en':
                        game.canvas.winfo_width=lambda:960
                        game.canvas.winfo_height=lambda:660
                        game.draw()
                        verify_panel(game)
                        capture(game,'preview-detail-cache-min-en.png')
                        game.canvas.winfo_width=lambda:1260
                        game.canvas.winfo_height=lambda:854
                s.return_parent()
                game.draw()
            capture(game,f'preview-attention-overview-cached-{language}.png')
        # Same prefix with caching off: every position is newly computed.
        game.toggle_cache()
        s=advance_to_prefix(game)
        s.forward.execute(2)
        s.forward.execute(3)
        cp=next(cp for cp in checkpoints(s.group) if cp.index==4)
        s.x,s.y=cp.x,cp.y
        game.interact()
        game.draw()
        capture(game,'preview-detail-cache-off-en.png')
        # A fresh multi-token prefill with cache ON is distinct from cache-off recomputation.
        game.cache_enabled=True
        game.guided=False
        game.config=replace(game.config,context=game.config.context[:6])
        game.restart()
        s=game.session
        game.jump_to(4)
        game.interact()
        for cp in s.room_checkpoints[:3]:
            s.x,s.y=cp.x,cp.y
            game.interact()
            assert s.rooms.active.progress==cp.index+1
        assert s.rooms.active.result.details['phase']=='prefill'
        capture(game,'preview-detail-mask-prefill-en.png')
        print('Attention previews generated: both languages, GQA, real cached decoding and cache-off comparison.')
    finally:
        game.close()


if __name__=='__main__':
    main()

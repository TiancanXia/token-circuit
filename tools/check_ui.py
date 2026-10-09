"""Hidden Tk integration; actual desktop input verification is separate."""
import tkinter as tk
from tokenwalk.app import Game
from tokenwalk.simulation.maps import houses,checkpoints,EXIT
from tools.walkthrough import complete_child


def check_language(language):
    root=tk.Tk()
    root.withdraw()
    game=Game(root,animate=False,language=language)
    errors=[]
    root.report_callback_exception=lambda typ,value,tb:errors.append(str(value))
    root.update_idletasks()
    game.canvas.winfo_width=lambda:1260
    game.canvas.winfo_height=lambda:854
    game.draw()
    event=tk.Event()
    event.keysym='e'
    game.press(event)
    assert game.session.scene=='micro'
    game.release(event)
    game.blur()
    assert not game.active and not game.keys
    game.focus()
    s=game.session
    while s.scene!='birth':
        if s.scene=='macro':
            group=s.expected_group
            h=next(h for h in houses() if h.key==group.house)
            s.x,s.y=h.x,h.y+35
            game.interact()
            assert s.scene=='micro'
        for cp in checkpoints(s.group):
            s.x,s.y=cp.x,cp.y
            game.draw()
            if cp.index==s.forward.next_index:
                game.interact()
                complete_child(game)
                game.draw()
        if s.scene!='birth':
            s.x,s.y=EXIT
            game.interact()
            assert s.scene=='macro'
    game.draw()
    token=s.pending_token
    for _ in range(90):
        s.update(1/60)
    assert s.generation==1 and s.token==token and s.scene=='macro'
    assert s.forward.config.context==s.initial_config.context[:2]
    game.canvas.winfo_width=lambda:960
    game.canvas.winfo_height=lambda:660
    game.draw()
    # Complete the remaining prescribed sentence with one F press per round.
    phases=set()
    while s.scene!='done':
        event.keysym='f'
        game.press(event)
        game.release(event)
        assert s.scene=='flight'
        generation=s.generation
        for _ in range(1000):
            s.update(1/30)
            game.draw()
            if s.scene=='birth':
                phases.add(s.transfer.phase)
            if s.scene=='done' or s.generation>generation:
                break
        else:
            raise AssertionError('Fast forward did not finish')
        assert s.forward.config.context==s.initial_config.context[:s.generation+1]
    assert phases=={'dissolve','drift','assemble'},phases
    assert s.generation==len(s.initial_config.context)-1
    assert len(s.generated)==len(s.initial_config.context)-1
    # Reposition the final prefix through every node without revealing another token.
    generation=s.generation
    cells=s.sentence_cells
    for index in range(len(s.forward.stages)):
        game.jump_to(index)
        assert s.forward.next_index==index and s.near_checkpoint.index==index
        game.toggle_navigation()
        assert game.navigation.selected==index
        game.draw()
        event.keysym='Escape'
        game.press(event)
        game.release(event)
        assert game.navigation is None and not game.closed
        game.interact()
        complete_child(game)
        assert s.generation==generation and s.sentence_cells==cells
    assert not errors,errors
    game.close()
    print(f'Tk integration passed ({language}): manual route, full quote, F, particle phases, final stop, all stage jumps/pickers and minimum window.')


def main():
    for language in ('zh','en'):
        check_language(language)


if __name__=='__main__':
    main()

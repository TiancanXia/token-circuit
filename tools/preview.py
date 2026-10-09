"""Optional Pillow reconstruction of Canvas output, not a desktop screenshot."""
from pathlib import Path
from functools import lru_cache
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
from tokenwalk.app import Game
from tokenwalk.simulation.maps import houses, checkpoints, EXIT
from tools.walkthrough import complete_child


OUTPUT=Path(__file__).resolve().parents[1]/'assets'/'previews'


@lru_cache(maxsize=64)
def preview_font(mono,size):
    return ImageFont.truetype('C:/Windows/Fonts/consola.ttf' if mono else 'C:/Windows/Fonts/msyh.ttc',size)


def capture(game,name=None):
    c=game.canvas
    image=Image.new('RGB',(c.winfo_width(),c.winfo_height()),'#101e2c')
    draw=ImageDraw.Draw(image)
    for item in c.find_all():
        kind,xy,fill=c.type(item),c.coords(item),c.itemcget(item,'fill')
        if not fill:
            continue
        if kind=='rectangle':
            if xy[2]>xy[0] and xy[3]>xy[1]:
                draw.rectangle((xy[0],xy[1],xy[2]-1,xy[3]-1),fill=fill)
        elif kind=='oval':
            draw.ellipse(xy,fill=fill)
        elif kind=='polygon':
            draw.polygon(list(zip(xy[::2],xy[1::2])),fill=fill)
        elif kind=='line':
            draw.line(list(zip(xy[::2],xy[1::2])),fill=fill,width=max(1,round(float(c.itemcget(item,'width')))))
        elif kind=='text':
            face,size=c.tk.splitlist(c.itemcget(item,'font'))[:2]
            f=preview_font(face==game.renderer.mono,abs(int(size)))
            value=c.itemcget(item,'text')
            box=c.bbox(item)
            # Respect Tk's wrapping by breaking into measured-width lines.
            width=float(c.itemcget(item,'width') or 0)
            lines=[]
            for paragraph in value.split('\n'):
                line=''
                for ch in paragraph:
                    if width and line and draw.textlength(line+ch,font=f)>width:
                        lines.append(line)
                        line=''
                    line+=ch
                lines.append(line)
            value='\n'.join(lines)
            if len(lines)>1:
                draw.multiline_text((box[0],box[1]),value,font=f,fill=fill,spacing=0)
            else:
                anchor=c.itemcget(item,'anchor')
                text_width=draw.textlength(value,font=f)
                x=xy[0]-(text_width if anchor=='ne' else text_width/2 if anchor=='n' else 0)
                draw.text((x,xy[1]),value,font=f,fill=fill,anchor='lt')
    if name:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        image.save(OUTPUT/name)
    return image


def verify_panel(game):
    c=game.canvas
    assert len(c.find_withtag('sentence-cell'))==len(game.session.sentence_cells)
    assert len(c.find_withtag('sentence-character'))==sum(ch is not None for ch in game.session.sentence_cells)
    for item in c.find_withtag('sentence-cell')+c.find_withtag('sentence-character'):
        box=c.bbox(item)
        assert 0<=box[0]<box[2]<=c.winfo_width(),box
        assert 0<=box[1]<box[3]<=c.winfo_height(),box


def main():
    root=tk.Tk()
    root.withdraw()
    game=Game(root,animate=False)
    game.canvas.winfo_width=lambda:1260
    game.canvas.winfo_height=lambda:854
    game.draw()
    capture(game,'preview-main.png')
    s=game.session
    for _ in range(40):
        s.update(1/60)
    game.draw()
    capture(game,'preview-main.png')
    while s.scene!='birth':
        h=next(h for h in houses() if h.key==s.expected_group.house)
        s.x,s.y=h.x,h.y+35
        game.interact()
        for cp in checkpoints(s.group):
            s.x,s.y=cp.x,cp.y
            game.interact()
            complete_child(game)
            verify_panel(game)
            kind=s.forward.stages[cp.index].kind
            if kind in ('qkv','attention') and s.group.block==0:
                capture(game,'preview-'+kind+'-world.png')
            if kind=='experts':
                capture(game,'preview-moe-world.png')
            game.canvas.winfo_width=lambda:960
            game.canvas.winfo_height=lambda:660
            game.draw()
            verify_panel(game)
            game.canvas.winfo_width=lambda:1260
            game.canvas.winfo_height=lambda:854
        if s.scene!='birth':
            s.x,s.y=EXIT
            game.interact()
        game.draw()
    capture(game,'preview-birth.png')
    for _ in range(90):
        s.update(1/60)
    game.draw()
    verify_panel(game)
    capture(game,'preview-next-character.png')
    # A single real forward, sampled at 15fps, including all transfer phases.
    from tokenwalk.simulation.world import Session
    game.session=Session(game.config,game.seed,guided=game.guided,language=game.language)
    s=game.session
    game.canvas.winfo_width=lambda:900
    game.canvas.winfo_height=lambda:610
    frames=[]
    s.start_fast_forward()
    flight_saved=False
    phases=set()
    for _ in range(150):
        s.update(1/15)
        game.draw()
        frames.append(capture(game))
        if s.scene=='flight' and s.forward.next_index>=4 and not flight_saved:
            capture(game,'preview-flight.png')
            flight_saved=True
        if (s.scene=='birth' and s.transfer.phase not in phases
                and s.transfer.progress>={'dissolve':.12,'drift':.47,'assemble':.88}[s.transfer.phase]):
            phases.add(s.transfer.phase)
            capture(game,'preview-'+s.transfer.phase+'.png')
        if s.generation==1:
            break
    frames[0].save(OUTPUT/'preview-flight-transfer.gif',save_all=True,append_images=frames[1:],duration=67,loop=0)
    assert phases=={'dissolve','drift','assemble'},phases
    # English uses the same scenes, including a real full-prefix attention pass.
    game.toggle_language()
    game.canvas.winfo_width=lambda:1260
    game.canvas.winfo_height=lambda:854
    game.draw()
    capture(game,'preview-main-en.png')
    s=game.session
    while s.scene!='done':
        s.start_fast_forward()
        generation=s.generation
        for _ in range(1000):
            s.update(1/30)
            if s.scene=='done' or s.generation>generation:
                break
        else:
            raise AssertionError('English forward did not finish')
    game.draw()
    verify_panel(game)
    capture(game,'preview-complete-en.png')
    from tokenwalk.simulation.maps import group_for
    s.group=next(group_for(s.forward.stages,i) for i,stage in enumerate(s.forward.stages)
                 if stage.kind=='attention' and stage.block==0)
    s.scene='micro'
    cp=next(cp for cp in checkpoints(s.group) if s.forward.stages[cp.index].kind=='attention')
    s.x,s.y=cp.x,cp.y
    game.draw()
    capture(game,'preview-attention-world-en.png')
    game.canvas.winfo_width=lambda:960
    game.canvas.winfo_height=lambda:660
    game.draw()
    verify_panel(game)
    capture(game,'preview-attention-min-en.png')
    for language in ('en','zh'):
        if game.language!=language:
            game.toggle_language()
        game.canvas.winfo_width=lambda:1260
        game.canvas.winfo_height=lambda:854
        for kind in ('unembedding','softmax'):
            index=next(i for i,stage in enumerate(game.session.forward.stages) if stage.kind==kind)
            game.jump_to(index)
            game.interact()
            capture(game,f'preview-{kind}-{language}.png')
        index=next(i for i,stage in enumerate(game.session.forward.stages)
                   if stage.block==1 and stage.kind=='router')
        game.jump_to(index)
        game.toggle_navigation()
        capture(game,f'preview-navigation-{language}.png')
        game.canvas.winfo_width=lambda:960
        game.canvas.winfo_height=lambda:660
        game.draw()
        capture(game,f'preview-navigation-min-{language}.png')
        game.toggle_navigation()
    game.close()
    print('Bilingual Canvas previews rendered; token masking/bounds verified at 1260x854 and 960x660.')

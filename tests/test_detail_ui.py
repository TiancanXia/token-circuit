import re
import tkinter as tk
import unittest
from dataclasses import replace
from tokenwalk.app import Game


class DetailInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk()
        self.root.withdraw()
        self.g=Game(self.root,animate=False)
        self.g.canvas.winfo_width=lambda:960
        self.g.canvas.winfo_height=lambda:660

    def tearDown(self):
        self.g.close()

    def key(self,name,release=True):
        e=tk.Event()
        e.keysym=name
        self.g.press(e)
        if release:
            self.g.release(e)

    def test_every_child_node_bilingual_and_bounded(self):
        g=self.g
        for language in ('zh','en'):
            if g.language!=language:
                g.toggle_language()
            g.toggle_attention()
            for index in (2,3,4,5,13,14,15,16):
                g.jump_to(index)
                g.interact()
                s=g.session
                self.assertEqual(s.scene,'detail')
                for cp in s.room_checkpoints:
                    s.x,s.y=cp.x,cp.y
                    g.interact()
                    g.draw()
                    c=g.canvas
                    for item in c.find_withtag('detail-number')+c.find_withtag('detail-explanation'):
                        box=c.bbox(item)
                        self.assertTrue(0<=box[0]<box[2]<=960,box)
                        self.assertTrue(0<=box[1]<box[3]<=660,box)
                    texts=[c.itemcget(i,'text') for i in c.find_all() if c.type(i)=='text']
                    if language=='en':
                        self.assertFalse(any(re.search('[\u4e00-\u9fff]',t) for t in texts),texts)
                    self.key('n')
                self.assertEqual(s.forward.next_index,index+1)
                self.key('Escape')
                self.assertEqual(s.scene,'micro')
                self.key('m')
                self.assertEqual(s.scene,'macro')

    def test_hold_e_and_m_cannot_skip_child_or_two_levels(self):
        g=self.g
        g.jump_to(2)
        self.key('e',False)
        self.key('e',False)
        self.assertEqual(g.session.scene,'detail')
        self.assertEqual(g.session.rooms.active.progress,0)
        self.assertEqual(g.session.forward.next_index,2)
        self.key('m',False)
        self.key('m',False)
        self.assertEqual(g.session.scene,'micro')

    def test_cropped_weighted_sum_includes_omitted_contribution(self):
        g=self.g
        g.guided=False
        g.config=replace(g.config,context=(0,1,2)*4)
        g.restart()
        g.jump_to(4)
        g.interact()
        s=g.session
        for cp in s.room_checkpoints:
            if cp.index>4:
                break  # Inspect weighted V before the separate Concat checkpoint.
            s.x,s.y=cp.x,cp.y
            g.interact()
        texts=[g.canvas.itemcget(i,'text') for i in g.canvas.find_all() if g.canvas.type(i)=='text']
        output=s.rooms.active.result.details['traces'][0]['outputs'][-1][0]
        formula=next(t for t in texts if '省略贡献' in t)
        self.assertIn(f'= {output:+.4f}',formula)

    def test_h_c_reset_once_and_l_preserves_attention_settings(self):
        g=self.g
        self.key('h',False)
        seed=g.seed
        self.key('h',False)
        self.assertEqual(g.seed,seed)
        self.assertEqual(g.config.num_kv_heads,2)
        self.key('c',False)
        seed=g.seed
        self.key('c',False)
        self.assertEqual(g.seed,seed)
        self.assertFalse(g.session.cache_enabled)
        g.toggle_language()
        self.assertEqual(g.config.num_kv_heads,2)
        self.assertFalse(g.session.cache_enabled)

    def test_jump_modal_blocks_setting_changes_and_child_progress(self):
        g=self.g
        g.jump_to(2)
        g.interact()
        visit=g.session.rooms.active
        g.toggle_navigation()
        for key in ('e','h','c','n','m'):
            if key=='e':
                continue  # Enter/E intentionally selects a jump target.
            self.key(key)
        self.assertEqual(visit.progress,0)
        self.assertEqual(visit.head,0)
        self.assertEqual(g.config.num_kv_heads,4)
        self.assertTrue(g.cache_enabled)

    def test_cached_decode_shows_all_visible_instead_of_masking(self):
        g=self.g
        s=g.session
        s.start_fast_forward()
        for _ in range(1000):
            s.update(1/30)
            if s.generation==1:
                break
        g.jump_to(4)
        g.interact()
        self.assertEqual(s.rooms.active.spec.nodes[2],'All visible')
        for cp in s.room_checkpoints[:3]:
            s.x,s.y=cp.x,cp.y
            g.interact()
        texts=[g.canvas.itemcget(i,'text') for i in g.canvas.find_all() if g.canvas.type(i)=='text']
        self.assertIn(s.tr('mask_bypass'),texts)
        self.assertFalse(g.canvas.find_withtag('mask-hidden'))

    def test_prefill_shows_actual_triangular_mask_and_correct_phase(self):
        g=self.g
        g.guided=False
        g.config=replace(g.config,context=(0,1,2))
        g.restart()
        g.jump_to(4)
        g.interact()
        texts=[g.canvas.itemcget(i,'text') for i in g.canvas.find_all() if g.canvas.type(i)=='text']
        self.assertTrue(any('预填充' in t for t in texts),texts)
        self.assertFalse(any('只计算新 token' in t for t in texts),texts)
        s=g.session
        self.assertEqual(s.rooms.active.spec.nodes[2],'Causal mask')
        for cp in s.room_checkpoints[:3]:
            s.x,s.y=cp.x,cp.y
            g.interact()
        self.assertEqual(len(g.canvas.find_withtag('mask-hidden')),3)
        self.assertEqual(len(g.canvas.find_withtag('mask-cell')),9)

    def test_concat_lives_once_in_head_attention_before_output_projection(self):
        g=self.g
        g.jump_to(4)
        g.interact()
        self.assertEqual(g.session.rooms.active.spec.nodes[-1],'Concat')
        g.jump_to(5)
        g.interact()
        self.assertEqual(g.session.rooms.active.spec.nodes[0],'Input')
        self.assertNotIn('Concat',g.session.rooms.active.spec.nodes)

    def test_unprocessed_node_does_not_substitute_previous_diagram(self):
        g=self.g
        g.jump_to(3)  # QKV has run; the selected RoPE has not.
        texts=[g.canvas.itemcget(i,'text') for i in g.canvas.find_all() if g.canvas.type(i)=='text']
        self.assertTrue(any('RoPE' in t and '尚未处理' in t for t in texts),texts)
        self.assertNotIn('Q  1×8',texts)

import unittest
from tokenwalk.simulation.world import Session
from tokenwalk.simulation.maps import houses, checkpoints, EXIT
from tokenwalk.model.architecture import ModelConfig


def walk(session, x, y):
    for _ in range(2000):
        dx,dy=x-session.x,y-session.y
        if abs(dx)<2 and abs(dy)<2:
            return
        session.update(1/60,(dx>2)-(dx<-2),(dy>2)-(dy<-2))
    raise AssertionError(f'Unreachable {x,y} from {session.x,session.y}')


def complete_round(s, physically_walk=False):
    last_house=None
    while s.scene not in ('birth','done'):
        group=s.expected_group
        house=next(h for h in houses() if h.key==group.house)
        if physically_walk and last_house:
            walk(s,house.x,house.y+35)
        else:
            s.x,s.y=house.x,house.y+35
        s.interact()
        if s.scene!='micro':
            raise AssertionError(s.message)
        for cp in checkpoints(group):
            if physically_walk:
                walk(s,cp.x,cp.y)
            else:
                s.x,s.y=cp.x,cp.y
            s.interact()
            if s.scene=='detail':
                for child in s.room_checkpoints:
                    if physically_walk:
                        walk(s,child.x,child.y)
                    else:
                        s.x,s.y=child.x,child.y
                    s.interact()
                if physically_walk:
                    walk(s,*EXIT)
                else:
                    s.x,s.y=EXIT
                s.interact()
                if s.scene!='micro':
                    raise AssertionError(s.message)
        if s.scene not in ('birth','done'):
            if physically_walk:
                walk(s,*EXIT)
            else:
                s.x,s.y=EXIT
            s.interact()
            if s.scene!='macro':
                raise AssertionError(s.message)
        last_house=house


class WorldTests(unittest.TestCase):
    def test_guided_sentence_starts_with_only_first_character(self):
        s=Session(ModelConfig(),guided=True)
        self.assertEqual(s.token,'风')
        self.assertEqual(s.forward.config.context,(0,))
        self.assertEqual(s.sentence_cells,('风',None,None))
        s.interact()
        s.interact()
        self.assertEqual(s.sentence_cells,('风',None,None))

    def test_guided_sentence_reveals_once_per_round_and_stops_at_end(self):
        s=Session(ModelConfig(),guided=True)
        complete_round(s,physically_walk=True)
        self.assertEqual(s.sentence_cells,('风',None,None))
        self.assertEqual(s.pending_token,'在')
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(s.sentence_cells,('风','在',None))
        self.assertEqual(s.token,'在')
        self.assertEqual(s.forward.config.context,(0,1))
        complete_round(s)
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(s.sentence_cells,('风','在','停'))
        complete_round(s)
        self.assertEqual(s.scene,'done')
        for _ in range(100):
            s.update(1/60,1,1)
        self.assertFalse(s.interact())
        self.assertEqual(s.generation,2)
        self.assertEqual(s.forward.config.context,(0,1,2))
        self.assertEqual(s.generated,['在','停'])

    def test_e_enters_room_and_executes_immediately(self):
        s=Session(ModelConfig())
        self.assertTrue(s.interact())
        self.assertEqual(s.scene,'micro')
        self.assertEqual(s.forward.next_index,0)
        self.assertTrue(s.interact())
        self.assertEqual(s.forward.next_index,1)
        self.assertIn('完成',s.message)
        self.assertFalse(s.interact())
        self.assertEqual(s.forward.next_index,1)

    def test_e_in_wrong_place_explains_what_to_do(self):
        s=Session(ModelConfig())
        s.x,s.y=500,245
        self.assertFalse(s.interact())
        self.assertIn('Embedding',s.message)
        self.assertEqual(s.scene,'macro')

    def test_full_two_block_path_is_walkable_and_output_restarts(self):
        s=Session(ModelConfig())
        original=s.forward.config.context
        complete_round(s,physically_walk=True)
        sampled=s.forward.sampled
        idx=s.forward.config.vocab.index(sampled)
        self.assertEqual(s.forward.next_index,27)
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(s.scene,'macro')
        self.assertEqual(s.generation,1)
        self.assertEqual(s.forward.next_index,0)
        self.assertEqual(s.forward.config.context,original+(idx,))
        self.assertEqual(s.token,sampled)
        self.assertEqual(len(s.generated),1)
        self.assertEqual(s.near_house.key,'embedding')
        complete_round(s)
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(s.generation,2)
        self.assertEqual(len(s.forward.config.context),len(original)+2)

    def test_returning_mid_attention_preserves_executed_stage(self):
        s=Session(ModelConfig())
        s.forward.execute(0)
        s.forward.execute(1)
        s.x,s.y=500,245
        s.interact()
        s.interact()
        from test_detail_rooms import finish_detail
        finish_detail(s)
        self.assertEqual(s.forward.next_index,3)
        s.return_macro()
        s.interact()
        self.assertEqual(s.scene,'micro')
        self.assertEqual(s.next_checkpoint.index,3)
        self.assertEqual(s.forward.next_index,3)

    def test_long_context_uses_labeled_bounded_window(self):
        config=ModelConfig(context=(0,1,2)*22)
        s=Session(config)
        self.assertEqual(len(s.forward.config.context),64)
        complete_round(s)
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(len(s.forward.config.context),64)
        self.assertEqual(len(s.generated),1)

    def test_completed_layers_can_be_revisited_without_reexecution(self):
        s=Session(ModelConfig())
        while s.forward.stages[s.forward.next_index].kind!='norm_final':
            s.forward.execute(s.forward.next_index)
        s.x,s.y=500,401
        s.interact()
        self.assertEqual(s.group.block,1)
        index=s.forward.next_index
        self.assertTrue(s.cycle_room_layer())
        self.assertEqual(s.group.block,0)
        self.assertEqual(s.forward.stages[s.group.indices[0]].kind,'up')
        self.assertFalse(s.interact())
        self.assertEqual(s.forward.next_index,index)
        self.assertTrue(s.cycle_room_layer())
        self.assertEqual(s.group.block,1)

    def test_layer_selector_cannot_reveal_future_layer(self):
        s=Session(ModelConfig())
        s.forward.execute(0)
        s.x,s.y=314,245
        s.interact()
        self.assertFalse(s.cycle_room_layer())
        self.assertEqual(s.group.block,0)


if __name__=='__main__':
    unittest.main()

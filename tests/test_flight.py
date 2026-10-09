import unittest
from tokenwalk.model.architecture import ModelConfig
from tokenwalk.model.forward import ForwardPass
from tokenwalk.simulation.maps import houses, macro_floor
from tokenwalk.simulation.world import Session


def finish_flight(session):
    for _ in range(2000):
        if session.scene!='flight':
            return
        session.update(1/60,1,1)
        if session.scene=='flight':
            if not macro_floor(session.x,session.y):
                raise AssertionError(f'Flight left the route at {session.x,session.y}')
    raise AssertionError('Flight never completed')


class FlightTests(unittest.TestCase):
    def test_starting_at_platform_edges_or_roads_never_crosses_a_gap(self):
        for position in ((680,477),(500,323),(500,477),(680,329),(110,290),(405,245),(314,310)):
            with self.subTest(position=position):
                s=Session(ModelConfig(),guided=True)
                self.assertTrue(s.can_walk(*position))
                s.x,s.y=position
                s.start_fast_forward()
                finish_flight(s)
                self.assertEqual(s.scene,'birth')

    def test_one_flight_runs_all_real_stages_and_repeated_blocks(self):
        s=Session(ModelConfig())
        self.assertTrue(s.start_fast_forward())
        self.assertEqual(s.scene,'flight')
        self.assertEqual(s.forward.next_index,0)
        finish_flight(s)
        self.assertEqual(s.scene,'birth')
        self.assertEqual(len(s.forward.results),27)
        manual=ForwardPass(ModelConfig(),s.seed)
        for i in range(len(manual.stages)):
            manual.execute(i)
        self.assertEqual([r.values for r in s.forward.results],[r.values for r in manual.results])
        self.assertEqual(s.forward.sampled,manual.sampled)
        self.assertEqual(s.generated,[])
        self.assertEqual(s.flight.visited,[('embedding',-1),('norm_attn',0),('attention',0),
                          ('residual_attn',0),('norm_ffn',0),('ffn',0),('residual_ffn',0),
                          ('norm_attn',1),('attention',1),('residual_attn',1),('norm_ffn',1),
                          ('ffn',1),('residual_ffn',1),('norm_final',-1),('unembedding',-1),('output',-1)])

    def test_flight_resumes_mid_room_without_repeating_results(self):
        s=Session(ModelConfig())
        s.forward.execute(0)
        s.forward.execute(1)
        s.x,s.y=500,245
        s.interact()
        s.interact()
        saved=s.forward.results[:]
        self.assertTrue(s.start_fast_forward())
        self.assertEqual(s.scene,'flight')
        finish_flight(s)
        self.assertEqual(len(s.forward.results),27)
        self.assertTrue(all(s.forward.results[i] is result for i,result in enumerate(saved)))

    def test_busy_actions_cannot_interrupt_or_repeat_flight(self):
        s=Session(ModelConfig(),guided=True)
        s.start_fast_forward()
        self.assertFalse(s.start_fast_forward())
        self.assertFalse(s.interact())
        self.assertFalse(s.return_macro())
        self.assertEqual(s.scene,'flight')
        finish_flight(s)
        self.assertFalse(s.start_fast_forward())
        self.assertEqual(s.generation,0)
        for _ in range(100):
            s.update(1/60)
        self.assertEqual(s.generation,1)
        self.assertEqual(s.token,'在')
        self.assertEqual(s.generated,['在'])
        self.assertEqual(s.scene,'macro')

    def test_transfer_moves_from_output_to_embedding_and_reveals_only_at_end(self):
        s=Session(ModelConfig(),guided=True)
        s.start_fast_forward()
        finish_flight(s)
        start=next(h for h in houses() if h.key=='output')
        end=next(h for h in houses() if h.key=='embedding')
        self.assertEqual(s.transfer.source,(start.x,start.y+35))
        self.assertEqual(s.transfer.target,(end.x,end.y+35))
        s.update(.1)
        self.assertEqual(s.transfer.phase,'dissolve')
        for _ in range(5):
            s.update(.1)
        self.assertEqual(s.transfer.phase,'drift')
        self.assertEqual(s.sentence_cells,('风',None,None))
        for _ in range(6):
            s.update(.1)
        self.assertEqual(s.transfer.phase,'assemble')
        for _ in range(4):
            s.update(.1)
        self.assertEqual(s.scene,'macro')
        self.assertEqual((s.x,s.y),s.transfer.target)
        self.assertEqual(s.sentence_cells,('风','在',None))

    def test_final_character_stops_without_spawning_an_extra_token(self):
        s=Session(ModelConfig(context=(0,)),guided=True)
        s.start_fast_forward()
        finish_flight(s)
        self.assertEqual(s.scene,'done')
        self.assertFalse(s.start_fast_forward())
        self.assertIsNone(s.transfer)
        self.assertEqual(s.generated,[])
        output=next(h for h in houses() if h.key=='output')
        self.assertEqual((s.x,s.y),(output.x,output.y+35))

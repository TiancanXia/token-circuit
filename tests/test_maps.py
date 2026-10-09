import unittest
from tokenwalk.simulation.maps import houses, group_for, checkpoints
from tokenwalk.model.architecture import ModelConfig, build_stages


class MapTests(unittest.TestCase):
    def test_collapsed_houses_cover_each_actual_stage_once(self):
        for blocks in (('dense','moe'),('moe','dense','dense')):
            stages=build_stages(ModelConfig(blocks=blocks))
            visited=[]
            index=0
            while index<len(stages):
                group=group_for(stages,index)
                self.assertEqual(group.indices[0],index)
                visited.extend(group.indices)
                self.assertTrue(any(h.key==group.house for h in houses()))
                index=group.indices[-1]+1
            self.assertEqual(visited,list(range(len(stages))))
            self.assertEqual(len(houses()),10)

    def test_attention_is_a_single_house_with_four_internal_stations(self):
        stages=build_stages(ModelConfig())
        group=group_for(stages,2)
        self.assertEqual(group.house,'attention')
        self.assertEqual([stages[i].kind for i in group.indices],['qkv','rope','attention','project'])
        self.assertEqual(len(checkpoints(group)),4)


if __name__=='__main__':
    unittest.main()

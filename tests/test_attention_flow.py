"""Attention semantics: head layout, visibility and a single packing boundary."""
import math
import unittest
from dataclasses import replace
from tokenwalk.model.architecture import ModelConfig, build_stages
from tokenwalk.model.forward import ForwardPass
from tokenwalk.model.mathops import attention_trace


class AttentionFlowTests(unittest.TestCase):
    def test_single_last_query_needs_no_explicit_causal_mask(self):
        q=[[1.,2.,3.,4.]]
        k=[[.1,.2,.3,.4],[.2,.3,.4,.5],[.3,.4,.5,.6]]
        _,traces=attention_trace(q,k,k,2,query_positions=[2])
        for trace in traces:
            self.assertEqual(trace['mask_applied'],[False])
            self.assertEqual(trace['scores'],trace['masked'])
            self.assertAlmostEqual(sum(trace['weights'][0]),1.)

    def test_cached_chunk_still_masks_future_new_tokens(self):
        q=[[1.,2.,3.,4.],[4.,3.,2.,1.]]
        k=[[.1,.2,.3,.4]]*4
        v=[[1.,2.,3.,4.]]*4
        output,traces=attention_trace(q,k,v,2,query_positions=[2,3])
        changed,_=attention_trace(q,k,v[:-1]+[[100.,200.,300.,400.]],2,query_positions=[2,3])
        self.assertEqual(output[0],changed[0])
        self.assertNotEqual(output[1],changed[1])
        for trace in traces:
            self.assertEqual(trace['mask_applied'],[True,False])
            self.assertEqual(trace['masked'][0][-1],-math.inf)
            self.assertEqual(trace['weights'][0][-1],0.)

    def test_mha_and_gqa_are_layouts_not_additional_stages(self):
        mha=ModelConfig(heads=4,context=(0,1,2))
        gqa=replace(mha,kv_heads=2)
        self.assertEqual(mha.attention_mode,'MHA')
        self.assertEqual(gqa.attention_mode,'GQA')
        self.assertEqual([mha.kv_head_for(h) for h in range(4)],[0,1,2,3])
        self.assertEqual([gqa.kv_head_for(h) for h in range(4)],[0,0,1,1])
        self.assertEqual([s.kind for s in build_stages(mha)],[s.kind for s in build_stages(gqa)])
        self.assertEqual(build_stages(mha)[4].title,'Head attention')
        self.assertEqual(build_stages(mha)[5].title,'Output · W_o')

    def test_all_query_head_outputs_are_packed_once_before_wo(self):
        for kv in (4,2,1):
            f=ForwardPass(ModelConfig(heads=4,kv_heads=kv))
            for index in range(6):
                f.execute(index)
            attention=f.results[4]
            expected=[x for trace in attention.details['traces'] for x in trace['outputs'][-1]]
            self.assertEqual(attention.values,expected)
            self.assertEqual(f.results[5].details['input'],expected)
            self.assertEqual(len(expected),f.config.dim)

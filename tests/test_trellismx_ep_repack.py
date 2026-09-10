"""EP preserves every intermediate partition while assigning disjoint experts."""
import unittest
import torch
from b12x.moe._shared.trellismx.p8_ep_repack import expert_partition_tensor
from b12x.moe._shared.trellismx.p8_coupled_scales import rank_local_coupled_signs


class ExpertPartitionTests(unittest.TestCase):
    def test_ep_partitions_reconstruct_all_physical_words(self):
        for name, shape, tp_axis, ep_axis in (
            ('w13_trellis', (2,12,8,16,80),3,1),
            ('w2_trellis', (12,16,8,80),1,0),
            ('w2_scale_ue8m0', (12,64,8),2,0)):
            full=torch.arange(torch.tensor(shape).prod()).reshape(shape)
            parents=list(full.chunk(4,tp_axis))
            for size in (2,4):
                got=[expert_partition_tensor(name,parents,r*12//size,(r+1)*12//size) for r in range(size)]
                self.assertTrue(torch.equal(torch.cat(got,ep_axis),full),name)

    def test_role_order_across_four_parents(self):
        for name,roles in (('w13_scale_ue8m0',2),('intermediate_scales_fp16',3)):
            full=torch.arange(12*roles*32*4).reshape(12,roles,32,4)
            parents=[p.reshape(12,roles*8,4) for p in full.chunk(4,2)]
            for size in (2,4):
                got=[expert_partition_tensor(name,parents,r*12//size,(r+1)*12//size) for r in range(size)]
                self.assertTrue(torch.equal(torch.cat(got,0),full.reshape(12,roles*32,4)),name)

    def test_replicated_vectors_and_bad_ranges_fail_closed(self):
        parents=[torch.ones(16) for _ in range(4)]
        self.assertTrue(torch.equal(expert_partition_tensor('gate_up_suh_fp16',parents,0,3),parents[0]))
        parents[3]+=1
        with self.assertRaises(ValueError):expert_partition_tensor('gate_up_suh_fp16',parents,0,3)
        with self.assertRaises(ValueError):expert_partition_tensor('w2_trellis',[torch.ones(12,16)]*4,11,13)
        with self.assertRaises(ValueError):expert_partition_tensor('unknown',[torch.ones(12)]*4,0,3)

    def test_full_width_signs_preserve_four_tp_slices(self):
        parents=[rank_local_coupled_signs(intermediate=512,rank=r) for r in range(4)]
        expected=torch.cat([p[:1024] for p in parents]+[p[1024:] for p in parents])
        self.assertTrue(torch.equal(expected,rank_local_coupled_signs(intermediate=2048,rank=0,world_size=1)))


if __name__=='__main__':unittest.main()

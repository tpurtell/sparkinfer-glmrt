"""GPU EP expert-slice comparison against all four original TP partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from b12x.moe._shared.trellismx.p8_native_kernel import P8NativeTPMoE

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path)
p.add_argument('--layer',type=int,default=3)
p.add_argument('--ep-size',type=int,choices=[2,4],required=True)
a=p.parse_args()
records=sorted([r for r in json.loads((a.checkpoint/'trellismx-manifest.json').read_text())['files'] if r['layer']==a.layer],key=lambda r:r['rank'])
paths=[a.checkpoint/r['path'] for r in records]; hashes=[r['sha256'] for r in records]
for path,digest in zip(paths,hashes):
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
common=dict(device=torch.device('cuda'),layer=a.layer,small_m_scheduler=True,fc1_tile_n=128,
 fuse_scratch_zero=True,grid_policy=True,fc1_warp_quant=False,fc1_broadcast_a=True,
 expected_design_sha256=records[0]['source_design_sha256'],
 expected_transform_sha256=hashlib.sha256((a.checkpoint/'design/transform.json').read_bytes()).hexdigest())
parents=[P8NativeTPMoE(path,tp_rank=r,world_size=4,intermediate=512,**common) for r,path in enumerate(paths)]
rank=a.ep_size-1;begin=rank*288//a.ep_size;count=288//a.ep_size
runtime=P8NativeTPMoE(tuple(paths),tp4_parent_sha256=tuple(hashes),tp_rank=0,world_size=1,
 intermediate=2048,ep_size=a.ep_size,ep_rank=rank,**common)
print(json.dumps({'gpu':torch.cuda.get_device_name(),'layer':a.layer,'ep_size':a.ep_size,'ep_rank':rank,'parents':hashes}),flush=True)
torch.manual_seed(20260910)
with torch.inference_mode():
 for m in (1,8,32,128,256):
  x=torch.randn(m,4096,dtype=torch.bfloat16,device='cuda')*.1
  local=torch.stack([torch.randperm(count,device='cuda')[:8] for _ in range(m)]).int()
  weights=torch.softmax(torch.randn(m,8,device='cuda'),dim=-1)
  expected=sum(r(x,weights,local+begin).float() for r in parents)
  for _ in range(2):actual=runtime(x,weights,local).clone()
  torch.cuda.synchronize()
  assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
  cosine=torch.nn.functional.cosine_similarity(actual.double().flatten(),expected.double().flatten(),dim=0).item()
  relative_l2=((actual.float()-expected).norm()/expected.norm()).item()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):captured=runtime(x,weights,local)
  for _ in range(3):graph.replay()
  torch.cuda.synchronize();torch.testing.assert_close(captured,actual,rtol=0,atol=0)
  print(json.dumps({'tokens':m,'cosine':cosine,'relative_l2':relative_l2,'graph_replay_exact':True}),flush=True)
  assert cosine>=.999 and relative_l2<=.02
print(json.dumps({'status':'passed','scope':'owned-expert numerical equivalence and graph replay only'}),flush=True)

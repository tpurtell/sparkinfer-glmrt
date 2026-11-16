"""Real Inkling PE checkpoint oracle: candidate uses native CuTe AOT boundaries.

The reference executes the pinned vendor tower modules; only vendor FA4 is
adapted to Torch SDPA with the same noncausal one-image geometry. This test
qualifies numerical math, not the unavailable vendor FA4 binary's reduction.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import types
import pytest
import torch
from ..conftest import require_b12x
from . import perception_vendor as vendor

VISION_SHA="34525c3b6afb44851de1f0341f60f6006760256c5e507b18cf5167a17497117c"


def _official(cfg):
    import numpy as np
    import torch.nn.functional as F
    from torch import nn
    from typing import Any,NamedTuple
    from collections.abc import Mapping,Iterable
    from collections import Counter
    path=Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"inkling_vision.py"
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VISION_SHA:raise ValueError("Inkling tower source changed")
    namespace=dict(torch=torch,np=np,F=F,nn=nn,Any=Any,NamedTuple=NamedTuple,Mapping=Mapping,
        Iterable=Iterable,Counter=Counter,__name__="inkling_pe_checkpoint_reference")
    official_ops=vendor.inkling()
    for name in ["apply_rotary_qk_","embed_tokens","scaled_residual_norm_"]:
        namespace[name]=getattr(official_ops,name)
    # Patch preprocessing is independently qualified by the pixel primitive;
    # this full-transformer oracle begins at normalized native patch rows.
    def flash(q,k,v,**kwargs):
        assert kwargs["causal"] is False
        out=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],scale=kwargs["softmax_scale"])
        return out[0].transpose(0,1).contiguous(),None
    namespace["flash_attn_varlen_func"]=flash
    tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["Inkling2VisionEncoder"](cfg["vision_config"],cfg["text_config"]["hidden_size"])


def _weights(snapshot):
    from safetensors import safe_open
    mapping=json.loads((snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    result={}
    for file in sorted({v for k,v in mapping.items() if k.startswith("model.visual.")}):
        with safe_open(snapshot/file,framework="pt",device="cpu") as source:
            for name in source.keys():
                if name.startswith("model.visual."):result[name[len("model.visual."):]]=source.get_tensor(name)
    assert len(result)==422
    return result


def _metric(name,actual,reference):
    a,r=actual.float(),reference.float()
    error=(a-r).abs();cos=torch.nn.functional.cosine_similarity(a.flatten()[None],r.flatten()[None]).item()
    rms=error.square().mean().sqrt().item()
    print(f"PE_CHECKPOINT {name} max_abs={error.max().item():.9g} rms={rms:.9g} cosine={cos:.12f} finite={bool(torch.isfinite(a).all())}",flush=True)
    assert torch.isfinite(a).all() and cos>=.999
    assert rms<=.025


@pytest.mark.parametrize("image_index",[0])
def test_inkling_full_checkpoint_native_transformer(image_index):
    require_b12x()
    from PIL import Image
    import numpy as np
    from b12x.integration.cuteafd.media import compile_biased_projection_aot,compile_patch_projection_aot,compile_gelu_aot,compile_vision_attention_aot
    from b12x.integration.cuteafd.perception import compile_scaled_residual_aot,compile_rotary_2d_aot,compile_merge2_aot
    from b12x.integration.cuteafd.perception_forward import compile_layer_norm_aot,compile_normalized_patches_aot,compile_learned_position_aot
    from cuda.bindings import runtime
    snapshot=Path(os.environ["CUTEAFD_PE_INKLING_SNAPSHOT"])
    cfg=json.loads((snapshot/"config.json").read_text());weights=_weights(snapshot)
    reference=_official(cfg).cuda().eval()
    state={}
    for name,value in weights.items():
        if ".self_attn." in name and any(f".{p}_proj." in name for p in "qkv"):continue
        state[name]=value.cuda()
    for i in range(23):
        for suffix in ["weight","bias"]:
            state[f"backbone.layers.{i}.self_attn.qkv_proj.{suffix}"]=torch.cat([weights[f"backbone.layers.{i}.self_attn.{p}_proj.{suffix}"] for p in "qkv"]).cuda()
    reference.load_state_dict(state,strict=True)
    paths=sorted(Path(os.environ["CUTEAFD_PE_REAL_IMAGES"]).glob("*.png"))
    # Use a genuine image crop without resize, bounding initial correctness
    # work while retaining natural pixels and sentinel edge padding.
    image=Image.open(paths[image_index]).convert("RGB").crop((0,0,197,157))
    rgb=torch.from_numpy(np.array(image)).cuda().float()
    h,w=160,200;gh,gw=h//20,w//20;n=gh*gw+1
    raw=torch.full((h,w,3),-1,device="cuda")
    raw[:157,:197]=rgb
    mean=torch.tensor([.48145466,.4578275,.40821073],device="cuda")
    ps=torch.tensor([.26862954,.2613026,.2757771],device="cuda")
    std=torch.tensor([.26862954,.26130258,.27577711],device="cuda")
    pixels=((raw/255-mean)/ps).bfloat16().float()
    pixels=(2*(pixels*std+mean)-1).bfloat16().float()
    patches=torch.empty(gh*gw,1200,device="cuda",dtype=torch.bfloat16)
    compile_normalized_patches_aot(patch=20,hwc=True).launch(pixels,patches,scalars=[h,w])
    plain=patches@state["backbone.embeddings.patch_embedding.weight"].T
    metadata=reference.metadata([[1,gh//2,gw//2]],torch.device("cuda",torch.cuda.current_device()))
    embeddings=reference.backbone.embeddings
    xref,_=vendor.inkling().embed_tokens(plain,metadata.patch_rows,embeddings.position_embedding.view(32,32,-1),metadata.meta,
        embeddings.class_embedding,embeddings.cls_position_embedding,reference.backbone.ln_pre,reference.backbone.layers[0].ln_1)
    projection=compile_patch_projection_aot(width=1200,outputs=1024)
    patch_emb=torch.empty_like(plain);projection.launch(patches,state["backbone.embeddings.patch_embedding.weight"],patch_emb,scalars=[gh*gw])
    x=torch.empty(n,1024,device="cuda",dtype=torch.bfloat16);tmp=torch.empty_like(x);normed=torch.empty_like(x)
    position=compile_learned_position_aot(hidden=1024,grid=32,class_token=True)
    position.launch(patch_emb,state["backbone.embeddings.position_embedding"],state["backbone.embeddings.class_embedding"],state["backbone.embeddings.cls_position_embedding"],tmp,scalars=[gh,gw])
    norm=compile_layer_norm_aot(hidden=1024,vendor_rsqrt=True)
    norm.launch(tmp,state["backbone.ln_pre.weight"],state["backbone.ln_pre.bias"],x,scalars=[n,1e-5])
    _metric("patch_pre",x,xref)
    qkv=torch.empty(n,3072,device="cuda",dtype=torch.bfloat16)
    capacity=16384
    q,k,v,o=[torch.empty(capacity,16,64,device="cuda",dtype=torch.bfloat16) for _ in range(4)]
    cu=torch.tensor([0,n],device="cuda",dtype=torch.int32);lse=torch.empty(16,capacity,device="cuda")
    freq=vendor.inkling()._rope_inv_freq(64,torch.device("cuda"))
    rotary=compile_rotary_2d_aot(heads=16,dim=64,class_token=True)
    attention=compile_vision_attention_aot(dim=64)
    status,l2=runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,0);assert int(status)==0
    qproj=compile_biased_projection_aot(width=1024,outputs=3072)
    oproj=compile_biased_projection_aot(width=1024,outputs=1024)
    up=compile_biased_projection_aot(width=1024,outputs=4096)
    down=compile_biased_projection_aot(width=4096,outputs=1024)
    gelu=compile_gelu_aot(width=4096);residual=compile_scaled_residual_aot(width=1024)
    delta=torch.empty_like(x);mlp=torch.empty(n,4096,device="cuda",dtype=torch.bfloat16)
    with torch.inference_mode():
        for i,layer in enumerate(reference.backbone.layers):
            base=f"backbone.layers.{i}."
            norm.launch(x,state[base+"ln_1.weight"],state[base+"ln_1.bias"],normed,scalars=[n,1e-5])
            qproj.launch(normed,state[base+"self_attn.qkv_proj.weight"],state[base+"self_attn.qkv_proj.bias"],qkv,scalars=[n])
            rotary.launch(qkv,freq,q,k,v,scalars=[n,gw])
            attention.launch(q,k,v,o,lse,cu,scalars=[64**-.5,l2])
            oproj.launch(o,state[base+"self_attn.out_proj.weight"],state[base+"self_attn.out_proj.bias"],delta,scalars=[n])
            residual.launch(x,delta,state[base+"ls_1.gamma"],tmp,scalars=[n]);x,tmp=tmp,x
            norm.launch(x,state[base+"ln_2.weight"],state[base+"ln_2.bias"],normed,scalars=[n,1e-5])
            up.launch(normed,state[base+"mlp.c_fc.weight"],state[base+"mlp.c_fc.bias"],mlp,scalars=[n]);gelu.launch(mlp,mlp,scalars=[n])
            down.launch(mlp,state[base+"mlp.c_proj.weight"],state[base+"mlp.c_proj.bias"],delta,scalars=[n])
            residual.launch(x,delta,state[base+"ls_2.gamma"],tmp,scalars=[n]);x,tmp=tmp,x
            layer(xref,layer.ln_1(xref),metadata,None)
            _metric(f"layer{i}",x,xref)
        units=gh*gw//4
        merged=torch.empty(units,4096,device="cuda",dtype=torch.bfloat16)
        compile_merge2_aot(width=1024,class_token=True).launch(x,merged,scalars=[gh,gw])
        adapted=torch.empty(units,4096,device="cuda",dtype=torch.bfloat16)
        final=torch.empty(units,4608,device="cuda",dtype=torch.bfloat16)
        compile_patch_projection_aot(width=4096,outputs=4096).launch(merged,state["proj_in.weight"],adapted,scalars=[units])
        gelu.launch(adapted,adapted,scalars=[units])
        compile_patch_projection_aot(width=4096,outputs=4608).launch(adapted,state["proj_out.weight"],final,scalars=[units])
        pooled=xref.index_select(0,metadata.pool_rows).reshape(units,4096)
        ref_final=reference.proj_out(torch.nn.functional.gelu(reference.proj_in(pooled)))
        _metric("projected_embeddings",final,ref_final)

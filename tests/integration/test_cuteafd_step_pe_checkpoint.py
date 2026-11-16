"""Actual Step-5 PE transformer and downsamplers against vendor-class reference.

Vendor module definitions are hash-pinned; TP linear/conv/attention adapters
express the canonical arithmetic at TP1, not library-specific runtime choices.
"""
from __future__ import annotations
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from ..conftest import require_b12x
from . import perception_vendor as vendor

SHA="348a7e4d5260d68232d91fe13a3b1a9a6badca4799457f1b7c8a1a8a112adcb2"


def _reference(cfg,mode):
    import torch.nn.functional as F
    from torch import nn
    from einops import rearrange,repeat
    class Linear(nn.Linear):
        def __init__(self,input,output,bias=True,**kwargs):super().__init__(input,output,bias=bias,dtype=torch.bfloat16)
        def forward(self,x):
            value=x.float()@self.weight.float().T
            if self.bias is not None:value=value+self.bias.float()
            return value.bfloat16(),None
    class QKV(Linear):
        def __init__(self,input,dim,heads,**kwargs):super().__init__(input,dim*heads*3,**kwargs)
    class Conv(nn.Conv2d):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs,dtype=torch.bfloat16)
        def forward(self,x):return F.conv2d(x.float(),self.weight.float(),None if self.bias is None else self.bias.float(),self.stride,self.padding).bfloat16()
    class Attention(nn.Module):
        def __init__(self,heads,dim,scale,**kwargs):super().__init__();self.heads,self.dim,self.scale=heads,dim,scale
        def forward(self,q,k,v):
            q,k,v=[x.reshape(x.shape[0],x.shape[1],self.heads,self.dim).transpose(1,2).float() for x in [q,k,v]]
            if mode[0]=="math":
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with sdpa_kernel(SDPBackend.MATH):out=F.scaled_dot_product_attention(q,k,v,scale=self.scale)
            else:
                scores=(q@k.transpose(-1,-2))*self.scale
                p=(scores-scores.max(-1,keepdim=True).values).exp()
                out=(p.bfloat16().float()@v)/p.sum(-1,keepdim=True)
            return out.transpose(1,2).flatten(-2).contiguous().bfloat16()
    path=Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"step_vl.py";raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA:raise ValueError("Step PE vendor source changed")
    scope=dict(torch=torch,nn=nn,F=F,rearrange=rearrange,repeat=repeat,
        ColumnParallelLinear=Linear,RowParallelLinear=Linear,QKVParallelLinear=QKV,Conv2dLayer=Conv,
        MMEncoderAttention=Attention,is_vit_use_data_parallel=lambda:True,get_tensor_model_parallel_world_size=lambda:1,
        _DEFAULT_NORM_LAYER=functools.partial(nn.LayerNorm,eps=1e-5,dtype=torch.bfloat16),__name__="step_pe_oracle")
    nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!="StepVLForConditionalGeneration"]
    import __future__
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec",flags=__future__.annotations.compiler_flag),scope)
    activation=lambda x:vendor.fused_quick_gelu()(x)
    return scope["PerceptionEncoder"](cfg,activation,norm_layer=functools.partial(nn.LayerNorm,eps=1e-5,dtype=torch.bfloat16))


def _load(snapshot):
    from safetensors import safe_open
    mapping=json.loads((snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    result={}
    for file in sorted({f for n,f in mapping.items() if n.startswith(("vision_model.","vit_large_projector."))}):
        with safe_open(snapshot/file,framework="pt",device="cpu") as weights:
            for name in weights.keys():
                if name.startswith(("vision_model.","vit_large_projector.")):result[name]=weights.get_tensor(name)
    assert len(result)==667
    return result


def _metrics(name,a,r):
    aa,rr=a.float(),r.float();rel=float(torch.linalg.vector_norm(aa-rr)/torch.linalg.vector_norm(rr).clamp_min(1e-30))
    cosine=float(torch.nn.functional.cosine_similarity(aa,rr,dim=-1).min())
    equal=int((a.contiguous().view(torch.int16)==r.contiguous().view(torch.int16)).sum())/a.numel()
    value=dict(name=name,relative_l2=rel,min_token_cosine=cosine,bit_equal=equal,max_abs=float((aa-rr).abs().max()))
    print("STEP_PE_CHECKPOINT "+json.dumps(value,sort_keys=True),flush=True)
    assert torch.isfinite(aa).all()
    return value


@pytest.mark.parametrize("side",[504])
def test_step_full_checkpoint_native_tower(side):
    require_b12x()
    from PIL import Image
    import numpy as np
    from b12x.integration.cuteafd import media,perception,perception_forward,image_resize
    from .test_cuteafd_image_resize import _coeff
    from cuda.bindings import runtime
    snapshot=Path(os.environ["CUTEAFD_PE_STEP5_SNAPSHOT"]);config=json.loads((snapshot/"config.json").read_text())
    settings=dict(config["vision_config"],use_abs_posemb=True,use_rope2d=True,use_ln_pre=True,mlp_ratio=8960/1536)
    cfg=SimpleNamespace(**settings);mode=["math"]
    old_tf32=torch.backends.cuda.matmul.allow_tf32;old_reduce=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    tensors=_load(snapshot);reference=_reference(cfg,mode).cuda().eval()
    state={}
    for name,tensor in tensors.items():
        if not name.startswith("vision_model."):continue
        name=name[len("vision_model."):]
        name=name.replace(".attn.in_proj_weight",".attn.qkv_proj.weight").replace(".attn.in_proj_bias",".attn.qkv_proj.bias")
        name=name.replace(".mlp.c_fc.",".mlp.fc1.").replace(".mlp.c_proj.",".mlp.fc2.")
        state[name]=tensor.cuda()
    reference.load_state_dict(state,strict=True)
    # Natural crop source; the official PIL crop source stage is separately
    # qualified. Exercise the actual504local tower grid first.
    files=list(Path(os.environ["CUTEAFD_PE_REAL_IMAGES"]).glob("*.jpg"))
    raw=Image.open(files[0]).convert("RGB").crop((0,0,504,504));rgb=torch.from_numpy(np.array(raw)).cuda()
    mean=torch.tensor([.48145466,.4578275,.40821073],device="cuda");std=torch.tensor([.26862954,.26130258,.27577711],device="cuda")
    pixels=torch.empty(side,side,3,device="cuda")
    image_resize.compile_rgb_normalize_aot().launch(rgb,mean,std,pixels,scalars=[side,side])
    gh=gw=side//14;n=gh*gw;units=n//16
    patches=torch.empty(n,588,device="cuda",dtype=torch.bfloat16)
    perception_forward.compile_normalized_patches_aot(patch=14).launch(pixels,patches,scalars=[side,side])
    conv_weight=state["conv1.weight"].reshape(1536,588)
    patch_emb=torch.empty(n,1536,device="cuda",dtype=torch.bfloat16)
    media.compile_patch_projection_aot(width=588,outputs=1536).launch(patches,conv_weight,patch_emb,scalars=[n])
    reference_patch=reference.conv1(pixels.permute(2,0,1)[None].bfloat16()).permute(0,2,3,1).reshape(n,1536)
    position=reference.sample_abs_posemb(gh,gw)[0]
    xref=reference.ln_pre(reference_patch+position);xref_class=xref.clone()
    tmp=torch.empty_like(xref);x=torch.empty_like(xref);normed=torch.empty_like(xref)
    dummy=torch.zeros(1536,device="cuda",dtype=torch.bfloat16)
    perception_forward.compile_learned_position_aot(hidden=1536,grid=52,torch_bilinear=True).launch(patch_emb,state["positional_embedding"],dummy,dummy,tmp,scalars=[gh,gw])
    norm=perception_forward.compile_layer_norm_aot(hidden=1536)
    norm.launch(tmp,state["ln_pre.weight"],state["ln_pre.bias"],x,scalars=[n,1e-5]);_metrics("patch_pre",x,xref)
    qkv=torch.empty(n,4608,device="cuda",dtype=torch.bfloat16)
    q,k,v,o=[torch.empty(16384,16,96,device="cuda",dtype=torch.bfloat16) for _ in range(4)]
    cu=torch.tensor([0,n],device="cuda",dtype=torch.int32);lse=torch.empty(16,16384,device="cuda")
    freq=vendor.step5().PerceptionEncoderRope2D(96,52,52)._compute_inv_freq(10000,48).cuda()
    rotary=perception.compile_rotary_2d_aot(heads=16,dim=96)
    attention=media.compile_vision_attention_aot(dim=96)
    status,l2=runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,0);assert int(status)==0
    qproj=media.compile_biased_projection_aot(width=1536,outputs=4608);oproj=media.compile_biased_projection_aot(width=1536,outputs=1536)
    up=media.compile_biased_projection_aot(width=1536,outputs=8960);down=media.compile_biased_projection_aot(width=8960,outputs=1536)
    gelu=perception.compile_quick_gelu_aot(width=8960);residual=perception.compile_scaled_residual_aot(width=1536)
    mlp=torch.empty(n,8960,device="cuda",dtype=torch.bfloat16);delta=torch.empty_like(x)
    def candidate(source,i):
        base=f"transformer.resblocks.{i}.";middle=torch.empty_like(source);out=torch.empty_like(source)
        norm.launch(source,state[base+"ln_1.weight"],state[base+"ln_1.bias"],normed,scalars=[n,1e-5])
        qproj.launch(normed,state[base+"attn.qkv_proj.weight"],state[base+"attn.qkv_proj.bias"],qkv,scalars=[n])
        rotary.launch(qkv,freq,q,k,v,scalars=[n,gw]);attention.launch(q,k,v,o,lse,cu,scalars=[96**-.5,l2])
        oproj.launch(o,state[base+"attn.out_proj.weight"],state[base+"attn.out_proj.bias"],delta,scalars=[n])
        residual.launch(source,delta,state[base+"ls_1.gamma"],middle,scalars=[n])
        norm.launch(middle,state[base+"ln_2.weight"],state[base+"ln_2.bias"],normed,scalars=[n,1e-5])
        up.launch(normed,state[base+"mlp.fc1.weight"],state[base+"mlp.fc1.bias"],mlp,scalars=[n]);gelu.launch(mlp,mlp,scalars=[n])
        down.launch(mlp,state[base+"mlp.fc2.weight"],state[base+"mlp.fc2.bias"],delta,scalars=[n])
        residual.launch(middle,delta,state[base+"ls_2.gamma"],out,scalars=[n]);return out
    native_teacher=[];class_teacher=[]
    with torch.inference_mode():
        for i,layer in enumerate(reference.transformer.resblocks):
            teacher=candidate(xref,i);x=candidate(x,i);class_forced=xref.clone()
            mode[0]="flash_class";class_forced=layer(class_forced[None],(gh,gw))[0];xref_class=layer(xref_class[None],(gh,gw))[0]
            mode[0]="math";xref=layer(xref[None],(gh,gw))[0]
            native_teacher.append(_metrics(f"teacher{i}",teacher,xref)["relative_l2"])
            class_teacher.append(_metrics(f"class_teacher{i}",class_forced,xref)["relative_l2"])
        # Actual native conv lowering/projection agrees with the checkpoint
        # conv module's canonical FP32accumulate output boundary.
        lower=torch.empty((n//4,1536*9),device="cuda",dtype=torch.bfloat16)
        c1=torch.empty(n//4,3072,device="cuda",dtype=torch.bfloat16)
        c2=torch.empty(units,6144,device="cuda",dtype=torch.bfloat16)
        perception.compile_conv3_stride2_lower_aot(channels=1536).launch(x,lower,scalars=[gh,gw])
        media.compile_biased_projection_aot(width=1536*9,outputs=3072).launch(lower,state["vit_downsampler1.weight"],state["vit_downsampler1.bias"],c1,scalars=[n//4])
        perception.compile_conv3_stride2_lower_aot(channels=3072).launch(c1,lower,scalars=[gh//2,gw//2])
        media.compile_biased_projection_aot(width=3072*9,outputs=6144).launch(lower,state["vit_downsampler2.weight"],state["vit_downsampler2.bias"],c2,scalars=[units])
        final=torch.empty(units,4096,device="cuda",dtype=torch.bfloat16);projector=tensors["vit_large_projector.weight"].cuda()
        media.compile_patch_projection_aot(width=6144,outputs=4096).launch(c2,projector,final,scalars=[units])
        def finish(source):
            source=source.T.reshape(1,1536,gh,gw)
            source=reference.vit_downsampler2(reference.vit_downsampler1(source))
            source=source[0].permute(1,2,0).reshape(units,6144)
            return (source.float()@projector.float().T).bfloat16()
        math_final=finish(xref);class_final=finish(xref_class)
        nm=_metrics("native_final_vs_math",final,math_final);cm=_metrics("class_final_vs_math",class_final,math_final)
        ratios=dict(teacher=max(native_teacher)/max(max(class_teacher),1e-30),
            final_relative=nm["relative_l2"]/max(cm["relative_l2"],1e-30),
            final_cosine_drift=(1-nm["min_token_cosine"])/max(1-cm["min_token_cosine"],1e-30))
        print("STEP_PE_ACCEPTANCE "+json.dumps(ratios,sort_keys=True),flush=True)
        assert all(value<=1.1 for value in ratios.values()),ratios
    torch.backends.cuda.matmul.allow_tf32=old_tf32;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_reduce

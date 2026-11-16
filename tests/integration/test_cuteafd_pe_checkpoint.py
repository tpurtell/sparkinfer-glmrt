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
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],scale=kwargs["softmax_scale"])
        return out[0].transpose(0,1).contiguous(),None
    namespace["flash_attn_varlen_func"]=flash
    def fp32_flash(q,k,v,**kwargs):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(q.float().transpose(0,1)[None],k.float().transpose(0,1)[None],v.float().transpose(0,1)[None],scale=kwargs["softmax_scale"])
        return out[0].transpose(0,1).contiguous().bfloat16(),None
    namespace["fp32_flash_attn_varlen_func"]=fp32_flash
    tree=ast.parse(raw)
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    # Model nn.Linear denotes one projection with a fused bias; library
    # heuristic BF16 partial/epilogue choices are context, not canonical math.
    class CanonicalLinear(nn.Linear):
        def forward(self,x):
            old_tf32=torch.backends.cuda.matmul.allow_tf32
            old_reduce=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
                value=x.float()@self.weight.float().T
                if self.bias is not None:value=value+self.bias.float()
                return value.bfloat16()
            finally:
                torch.backends.cuda.matmul.allow_tf32=old_tf32
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_reduce
    namespace["vendor_default_linear"]=namespace["_linear"]
    namespace["_linear"]=lambda input,output,bias=True:CanonicalLinear(input,output,bias=bias,dtype=torch.bfloat16)
    tower=namespace["Inkling2VisionEncoder"](cfg["vision_config"],cfg["text_config"]["hidden_size"])
    tower._reference_globals=namespace
    return tower


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


FINDINGS=[]


def _metric(name,actual,reference,*,teacher=False,final=False):
    a,r=actual.float(),reference.float()
    error=(a-r).abs()
    relative=torch.linalg.vector_norm(a-r)/torch.linalg.vector_norm(r).clamp_min(1e-30)
    cosine=torch.nn.functional.cosine_similarity(a,r,dim=-1).min().item()
    equal=int((actual.contiguous().view(torch.int16)==reference.contiguous().view(torch.int16)).sum())/actual.numel()
    result=dict(name=name,max_abs=error.max().item(),relative_l2=relative.item(),min_token_cosine=cosine,bit_equal=equal)
    print("PE_CHECKPOINT "+json.dumps(result,sort_keys=True),flush=True)
    assert torch.isfinite(a).all()
    if teacher and not (relative<=2e-3 and equal>=.85):FINDINGS.append(result)
    if final and not (relative<=1e-2 and cosine>=.9999):FINDINGS.append(result)
    return result


@pytest.mark.parametrize("image_index",[0])
def test_inkling_full_checkpoint_native_transformer(image_index):
    require_b12x()
    FINDINGS.clear()
    print("PE_REFERENCE_BACKENDS "+json.dumps(dict(bf16="SDPBackend.MATH",fp32="SDPBackend.MATH",projection="FP32 GEMM + FP32 bias + single BF16 store",
        device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)),flush=True)
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
    plain=reference.backbone.embeddings.patch_embedding(patches)
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
    def candidate_layer(source,i):
        base=f"backbone.layers.{i}."
        middle=torch.empty_like(source);output=torch.empty_like(source)
        norm.launch(source,state[base+"ln_1.weight"],state[base+"ln_1.bias"],normed,scalars=[n,1e-5])
        qproj.launch(normed,state[base+"self_attn.qkv_proj.weight"],state[base+"self_attn.qkv_proj.bias"],qkv,scalars=[n])
        rotary.launch(qkv,freq,q,k,v,scalars=[n,gw])
        attention.launch(q,k,v,o,lse,cu,scalars=[64**-.5,l2])
        oproj.launch(o,state[base+"self_attn.out_proj.weight"],state[base+"self_attn.out_proj.bias"],delta,scalars=[n])
        residual.launch(source,delta,state[base+"ls_1.gamma"],middle,scalars=[n])
        norm.launch(middle,state[base+"ln_2.weight"],state[base+"ln_2.bias"],normed,scalars=[n,1e-5])
        up.launch(normed,state[base+"mlp.c_fc.weight"],state[base+"mlp.c_fc.bias"],mlp,scalars=[n]);gelu.launch(mlp,mlp,scalars=[n])
        down.launch(mlp,state[base+"mlp.c_proj.weight"],state[base+"mlp.c_proj.bias"],delta,scalars=[n])
        residual.launch(middle,delta,state[base+"ls_2.gamma"],output,scalars=[n])
        return output
    def projection_arms(name,source,weight,bias,native,canonical):
        default=torch.nn.functional.linear(source,weight,bias)
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        old_reduce=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
            strict=torch.nn.functional.linear(source,weight,bias)
            fp32=(source.float()@weight.float().T+bias.float()).bfloat16()
            separate=(source@weight.T)+bias
            _metric(name+"_native_vs_canonical",native,canonical)
            _metric(name+"_native_vs_vendor_default",native,default)
            _metric(name+"_native_vs_strict",native,strict)
            _metric(name+"_native_vs_explicit_fp32",native,fp32)
            _metric(name+"_native_vs_separate_bf16_bias",native,separate)
            _metric(name+"_vendor_default_vs_strict",default,strict)
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_reduce
    def probe_layer0(source):
        layer=reference.backbone.layers[0];base="backbone.layers.0."
        def compare(stage,a,r):return _metric("substage0_"+stage,a,r)
        rnorm=layer.ln_1(source)
        norm.launch(source,state[base+"ln_1.weight"],state[base+"ln_1.bias"],normed,scalars=[n,1e-5]);compare("norm1",normed,rnorm)
        rqkv=layer.self_attn.qkv_proj(rnorm)
        qproj.launch(rnorm,state[base+"self_attn.qkv_proj.weight"],state[base+"self_attn.qkv_proj.bias"],qkv,scalars=[n]);compare("qkv",qkv,rqkv)
        print("PE_SUBSTAGE qk_norm=absent activation=exact_erf_GELU",flush=True)
        rotated=rqkv.clone();vendor.inkling().apply_rotary_qk_(rotated,metadata.meta,16)
        rotary.launch(rqkv,freq,q,k,v,scalars=[n,gw])
        rq,rk,rv=rotated.reshape(n,3,16,64).unbind(1)
        compare("rotary_q",q[:n],rq);compare("rotary_k",k[:n],rk)
        bf16=reference._reference_globals["flash_attn_varlen_func"]
        fp32=reference._reference_globals["fp32_flash_attn_varlen_func"]
        args=dict(causal=False,softmax_scale=64**-.5)
        rattention=bf16(rq,rk,rv,**args)[0];rattention32=fp32(rq,rk,rv,**args)[0]
        q[:n].copy_(rq);k[:n].copy_(rk);v[:n].copy_(rv)
        attention.launch(q,k,v,o,lse,cu,scalars=[64**-.5,l2])
        compare("attention_vs_bf16_math",o[:n],rattention);compare("attention_vs_fp32_math",o[:n],rattention32)
        compare("reference_bf16_vs_fp32_math",rattention,rattention32)
        rprojection=layer.self_attn.out_proj(rattention.reshape(n,1024))
        oproj.launch(rattention,state[base+"self_attn.out_proj.weight"],state[base+"self_attn.out_proj.bias"],delta,scalars=[n]);compare("o_projection",delta,rprojection)
        projection_arms("o_projection",rattention.reshape(n,1024),state[base+"self_attn.out_proj.weight"],state[base+"self_attn.out_proj.bias"],delta,rprojection)
        rmid=source.clone();vendor.inkling().scaled_residual_norm_(rmid,rprojection,layer.ls_1.gamma,None)
        residual.launch(source,rprojection,state[base+"ls_1.gamma"],tmp,scalars=[n]);compare("attention_scale_residual",tmp,rmid)
        rnorm2=layer.ln_2(rmid)
        norm.launch(rmid,state[base+"ln_2.weight"],state[base+"ln_2.bias"],normed,scalars=[n,1e-5]);compare("norm2",normed,rnorm2)
        rfc1=layer.mlp.c_fc(rnorm2)
        up.launch(rnorm2,state[base+"mlp.c_fc.weight"],state[base+"mlp.c_fc.bias"],mlp,scalars=[n]);compare("mlp_fc1",mlp,rfc1)
        rgelu=torch.nn.functional.gelu(rfc1)
        gelu.launch(rfc1,mlp,scalars=[n]);compare("gelu",mlp,rgelu)
        rfc2=layer.mlp.c_proj(rgelu)
        down.launch(rgelu,state[base+"mlp.c_proj.weight"],state[base+"mlp.c_proj.bias"],delta,scalars=[n]);compare("mlp_fc2",delta,rfc2)
        projection_arms("mlp_fc2",rgelu,state[base+"mlp.c_proj.weight"],state[base+"mlp.c_proj.bias"],delta,rfc2)
        rfinal=rmid.clone();vendor.inkling().scaled_residual_norm_(rfinal,rfc2,layer.ls_2.gamma,None)
        residual.launch(rmid,rfc2,state[base+"ls_2.gamma"],tmp,scalars=[n]);compare("mlp_scale_residual",tmp,rfinal)
    with torch.inference_mode():probe_layer0(xref)
    xref32=xref.clone();scope=reference._reference_globals
    bf16_attention=scope["flash_attn_varlen_func"]
    with torch.inference_mode():
        for i,layer in enumerate(reference.backbone.layers):
            teacher=candidate_layer(xref,i)
            x=candidate_layer(x,i)
            ref32_teacher=xref.clone()
            scope["flash_attn_varlen_func"]=scope["fp32_flash_attn_varlen_func"]
            layer(ref32_teacher,layer.ln_1(ref32_teacher),metadata,None)
            layer(xref32,layer.ln_1(xref32),metadata,None)
            scope["flash_attn_varlen_func"]=bf16_attention
            layer(xref,layer.ln_1(xref),metadata,None)
            _metric(f"teacher_layer{i}",teacher,xref,teacher=True)
            _metric(f"reference_attention_variance_layer{i}",ref32_teacher,xref)
            _metric(f"free_layer{i}",x,xref)
            _metric(f"free_vs_fp32_attention_layer{i}",x,xref32)
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
        pooled32=xref32.index_select(0,metadata.pool_rows).reshape(units,4096)
        ref_final32=reference.proj_out(torch.nn.functional.gelu(reference.proj_in(pooled32)))
        _metric("projected_embeddings_vs_bf16_attention",final,ref_final,final=True)
        _metric("projected_embeddings_vs_fp32_attention",final,ref_final32)
        _metric("reference_bf16_vs_fp32_attention",ref_final,ref_final32)
        assert not FINDINGS,FINDINGS

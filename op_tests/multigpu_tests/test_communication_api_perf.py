"""Representative communication API correctness and latency, run with torchrun.

HIP: GPU-event Graph samples. Iris/groups: synchronized eager wall and GPU time.
Every sample is retained, including outliers. No installed package is modified.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist


def stats(values):
    return dict(median_us=statistics.median(values),
                p95_us=sorted(values)[math.ceil(.95 * len(values)) - 1],
                max_us=max(values),
                cv_percent=100 * statistics.pstdev(values) / statistics.mean(values))


def close(actual, expected, tol=.03):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol)
    assert torch.isfinite(actual.float()).all()


class Bench:
    def __init__(self, args):
        self.args, self.rows = args, []
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.path = Path(args.output)
        self.path.mkdir(parents=True, exist_ok=True)

    def measure(self, name, shape, dtype, call, check, *, capture=None, eager=False,
                refresh=None, **meta):
        check(call())
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        dist.barrier()
        graph, output = None, None
        if not eager:
            with capture() if capture else nullcontext():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(self.args.graph_ops):
                        output = call()
            graph.replay()
            torch.cuda.synchronize()
            check(output)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        gpu, wall = [], []
        ops = 1 if eager else self.args.graph_ops
        for _ in range(self.args.samples):
            dist.barrier()
            before = time.perf_counter_ns()
            start.record()
            if eager:
                output = call()
            else:
                graph.replay()
            end.record()
            end.synchronize()
            wall.append((time.perf_counter_ns() - before) / 1000 / ops)
            gpu.append(start.elapsed_time(end) * 1000 / ops)
        check(output)
        if refresh is not None:
            # Validate new contents at the same input address, outside timing.
            refresh()
            if eager:
                output = call()
            else:
                graph.replay()
            torch.cuda.synchronize()
            check(output)
        collected = [None] * self.world
        dist.all_gather_object(collected, dict(gpu_us=gpu, wall_us=wall))
        worst_gpu = [max(r['gpu_us'][i] for r in collected) for i in range(self.args.samples)]
        worst_wall = [max(r['wall_us'][i] for r in collected) for i in range(self.args.samples)]
        row = dict(name=name, shape=list(shape), dtype=str(dtype).replace('torch.', ''),
                   tp=self.world, execution='eager' if eager else 'graph', ops_per_sample=ops,
                   samples=self.args.samples, gpu=stats(worst_gpu), wall=stats(worst_wall),
                   collective_gpu_us=worst_gpu, collective_wall_us=worst_wall,
                   per_rank=collected, correctness='pass', **meta)
        self.rows.append(row)
        if self.rank == 0:
            (self.path/'results.json').write_text(json.dumps(self.rows, indent=2))
            print('PERF_PASS', name, shape, str(dtype), json.dumps(row['wall' if eager else 'gpu']), flush=True)
        del graph, output
        torch.cuda.synchronize()
        dist.barrier()


def car_ar_suite(b):
    """Unquantized, out-of-place SUM versus clone + RCCL, with exact fixtures."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(dist.group.WORLD, torch.cuda.current_device(),
                         max_size=128 << 20, enable_register_for_capturing=True)
    assert not ca.disabled
    rccl = dist.new_group(backend='nccl')
    try:
        sizes = (4 << 10, 16 << 10, 24 << 10, 192 << 10,
                 512 << 10, 2 << 20, 8 << 20, 32 << 20)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for size_index, nbytes in enumerate(sizes):
                numel = nbytes // torch.empty((), dtype=dtype).element_size()
                base = (torch.arange(numel, device='cuda') % 8).to(dtype) / 8
                x = base + b.rank
                state = [0]

                def refresh():
                    state[0] += 1
                    x.copy_(base + b.rank + state[0] / 4)

                def check(y):
                    assert y is not None and y.data_ptr() != x.data_ptr()
                    expected = (base + state[0] / 4) * b.world + b.world * (b.world - 1) / 2
                    torch.testing.assert_close(y, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x, base + b.rank + state[0] / 4, rtol=0, atol=0)

                assert ca.should_custom_ar(x)

                def car():
                    return ca.custom_all_reduce(x, use_new=True, open_fp8_quant=False)

                def reference():
                    y = x.clone()
                    dist.all_reduce(y, group=rccl)
                    return y

                cases = [('car_all_reduce', car, ca.capture),
                         ('rccl_clone_all_reduce', reference, None)]
                # Alternate ordering without dropping slow samples.
                if size_index % 2:
                    cases.reverse()
                for eager in (False, True):
                    for name, call, capture in cases:
                        b.measure(name, x.shape, dtype, call, check,
                                  capture=capture, eager=eager, refresh=refresh,
                                  payload_bytes=nbytes, transport='ipc',
                                  output_allocation='included', input_clone=name.startswith('rccl'),
                                  register_capture_requested=True,
                                  register_capture_effective=ca.enable_register_for_capturing,
                                  use_new=True, open_fp8_quant=False)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        ca.close()
        dist.destroy_process_group(rccl)


def car_suite(b):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.ops import custom_all_reduce as native
    from aiter.ops import quick_all_reduce as qr
    ca = CustomAllreduce(dist.group.WORLD, torch.cuda.current_device(),
                         max_size=64 << 20, enable_register_for_capturing=False)
    assert not ca.disabled
    world, rank = b.world, b.rank
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for m, h in ((4, 2048), (128, 8192)):
                base = (torch.arange(m*h, device='cuda').reshape(m,h).float() % 17 - 8) / 16
                inputs = [(base + r / 8).to(dtype) for r in range(world)]
                x = inputs[rank]
                summed = sum(t.float() for t in inputs).to(dtype)
                for dim in (0, -1):
                    expected = torch.cat(inputs, dim)
                    assert ca.should_custom_ag(x, dim)
                    b.measure('car_ag_dim'+str(dim), x.shape, dtype,
                              lambda: ca.custom_all_gather(x, dim),
                              lambda y: close(y, expected, 0), capture=ca.capture)
                    shape = list(x.shape); shape[dim] //= world
                    output = torch.empty(shape, dtype=dtype, device='cuda')
                    expected = summed.chunk(world, dim)[rank]
                    assert ca.should_custom_rs(x, dim)
                    def rs():
                        ca.custom_reduce_scatter(x, output, dim)
                        return output
                    b.measure('car_rs_dim'+str(dim), x.shape, dtype, rs,
                              lambda y: close(y, expected), capture=ca.capture)
                mid = x.reshape(m, world, h//world)
                mid_out = torch.empty((m, 1, h//world), dtype=dtype, device='cuda')
                mid_ref = summed.reshape_as(mid)[:, rank:rank+1]
                def rs_mid():
                    ca.custom_reduce_scatter(mid, mid_out, 1)
                    return mid_out
                b.measure('car_rs_middle', mid.shape, dtype, rs_mid,
                          lambda y: close(y, mid_ref), capture=ca.capture)
                xi = torch.arange(m*h, device='cuda', dtype=torch.int64).reshape(m,h) + rank * 2**40
                expected_i = torch.cat([torch.arange(m*h, device='cuda', dtype=torch.int64).reshape(m,h)+r*2**40 for r in range(world)], -1)
                if dtype == torch.bfloat16:
                    b.measure('car_ag_int64', xi.shape, torch.int64,
                              lambda: ca.custom_all_gather(xi, -1),
                              lambda y: torch.testing.assert_close(y, expected_i, rtol=0, atol=0), capture=ca.capture)
                residual = torch.full_like(x, .125)
                weight = (torch.arange(h, device='cuda').float() % 7 / 8 + .5).to(dtype)
                z = summed.float() + residual.float()
                raw = z * torch.rsqrt(z.square().mean(-1, keepdim=True)+1e-6)
                for kind in ('norm','gemma','padding','packed','fp8_token','fp8_token_gemma','fp8_group','fp8_group_transpose'):
                    storage = torch.full((m,h+16),7.,dtype=dtype,device='cuda')
                    storage[:,:h].copy_(x)
                    inp = storage if kind=='padding' else storage[:,:h] if kind=='packed' else x
                    norm = raw * (weight.float()+(1 if kind in ('gemma','fp8_token_gemma') else 0))
                    def call():
                        if kind.startswith('fp8_token'):
                            return ca.custom_fused_ar_rms_quant(inp,residual,weight,1e-6,gemma_norm=kind.endswith('gemma'),emit_bf16=True)
                        if kind.startswith('fp8_group'):
                            return ca.custom_fused_ar_rms_per_group_quant(inp,residual,weight,1e-6,emit_bf16=True,transpose_scale=kind.endswith('transpose'))
                        if kind=='packed':
                            return ca.custom_fused_ar_rms_packed_input(inp,residual,weight,1e-6)
                        return ca.custom_fused_ar_rms(inp,residual,weight,1e-6,out_hidden_dim=h+32 if kind=='padding' else 0,gemma_norm=kind=='gemma')
                    def check(y):
                        close(y[1],z)
                        if kind.startswith('fp8'):
                            close(y[3],norm)
                            groups = h//128 if kind.startswith('fp8_group') else 1
                            scale = norm.reshape(m,groups,-1).abs().amax(-1)/448
                            close(y[2],scale)
                            dq=y[0].float().reshape(m,groups,-1)*y[2].unsqueeze(-1)
                            torch.testing.assert_close(dq.reshape_as(norm),norm,rtol=.1,atol=.03)
                            if kind.endswith('transpose'): assert y[2].stride()==(1,m)
                        else:
                            close(y[0][:,:h],norm)
                            if kind=='padding': assert not y[0][:,h:].count_nonzero()
                    b.measure('car_'+kind, inp.shape, dtype, call, check, capture=ca.capture,
                              hidden_dim=h, use_1stage=False, emit_mirror=kind.startswith('fp8'))
                # Exact codec fixture; its input distribution is recorded separately.
                values=torch.tensor([0,.25,.5,.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,5,6,-6],device='cuda')
                w4=values.repeat(h//16).to(dtype)
                x4=torch.full_like(x,1/world);r4=torch.zeros_like(x)
                bounds=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5],device='cuda')
                codes=torch.bucketize(w4.float().abs(),bounds,right=True).to(torch.uint8)
                codes|=(w4<0).to(torch.uint8)<<3
                packed=codes[::2]|(codes[1::2]<<4)
                def check4(y):
                    torch.testing.assert_close(y[0],packed.expand(m,-1),rtol=0,atol=0)
                    torch.testing.assert_close(y[2],torch.full_like(y[2],127),rtol=0,atol=0)
                    close(y[1],torch.ones_like(x4));close(y[3],w4.expand_as(x4))
                b.measure('car_mxfp4',x4.shape,dtype,
                          lambda:ca.custom_fused_ar_rms_mxfp4_quant(x4,r4,w4,0.,emit_bf16=True,use_1stage=True),
                          check4,capture=ca.capture,input_pattern='exact_e2m1_fixture',emit_mirror=True,use_1stage=True)

                qbase=(torch.arange(m*1024,device='cuda').reshape(m,1024).float()%31-15)/32
                qs=[(qbase+r/16).to(dtype) for r in range(world)]
                qkv=qs[rank];qw=torch.ones(512,device='cuda',dtype=dtype);kw=torch.ones(256,device='cuda',dtype=dtype)
                angle=torch.arange(32*32,device='cuda').reshape(32,32).float()/37
                cache=torch.cat((angle.cos(),angle.sin()),-1).to(dtype)
                pos=torch.arange(m,device='cuda',dtype=torch.int64)%32
                ref=[]
                for start,width in ((0,512),(512,256)):
                    var=sum(t[:,start:start+width].float().square().mean(-1,keepdim=True) for t in qs)/world
                    norm=(qkv[:,start:start+width].float()*torch.rsqrt(var+1e-6)).to(dtype).float()
                    heads=norm.reshape(m,-1,128);rotated=heads.clone()
                    c=cache[pos,:32].float()[:,None,:];s=cache[pos,32:].float()[:,None,:]
                    rotated[...,:32]=heads[...,:32]*c-heads[...,32:64]*s
                    rotated[...,32:64]=heads[...,32:64]*c+heads[...,:32]*s
                    ref.append(rotated.flatten(1))
                ref.append(qkv[:,768:])
                def check_qk(y):
                    for got,want in zip(y,ref):close(got,want)
                b.measure('qknorm_rope',qkv.shape,dtype,
                          lambda:ca.custom_fused_qknorm_ar_rope(qkv,qw,kw,cache,pos,128,64,1e-6),
                          check_qk,capture=ca.capture,head_dim=128,rotary_dim=64)

                mhc_h=2048 if m==4 else 4096
                mhc_x=x[:,:mhc_h].contiguous()
                mr=torch.full((m,4,mhc_h),.125,dtype=dtype,device='cuda')
                post=(torch.arange(m*4,device='cuda').float().reshape(m,4)%7+1)/16
                comb=(torch.arange(m*16,device='cuda').float().reshape(m,4,4)%11-5)/32
                mo=torch.empty_like(mr)
                mr_ref=summed[:,:mhc_h].float()[:,None,:]*post[:,:,None]+torch.einsum('mih,mij->mjh',mr.float(),comb)
                for path in ('only','one_stage','split'):
                    fn=getattr(native,'fused_allreduce_mhc_post_'+path)
                    def mhc():
                        fn(ca._ptr,mhc_x,mo,mr,post,comb,True,False,ca._pool['input'].data_ptr,ca._pool['input'].max_size)
                        return mo
                    b.measure('mhc_'+path,mhc_x.shape,dtype,mhc,lambda y:close(y,mr_ref))
        # Native QR is measured without the high-level payload admission threshold.
        ptr=qr.init_custom_qr(rank,world,64<<20)
        handles=[None]*world;dist.all_gather_object(handles,qr.qr_get_handle(ptr));qr.qr_open_handles(ptr,handles)
        try:
            limits={1:(.08,.20),2:(.08,.15),3:(.30,.50),4:(.60,.85)}
            for dtype in (torch.float16,torch.bfloat16):
                for m,h in ((4,2048),(128,8192)):
                    base=(torch.arange(m*h,device='cuda').reshape(m,h).float()%37-18)/32
                    inputs=[(base+r/32).to(dtype) for r in range(world)]
                    x=inputs[rank];ref=sum(t.float() for t in inputs)
                    out=torch.empty_like(x);res=torch.full_like(x,.125)
                    rout=torch.empty_like(x);norm=torch.empty_like(x);weight=torch.ones(h,device='cuda',dtype=dtype)
                    for level,name in enumerate(('fp','fp8','int6','int4','int3')):
                        if level==4 and world!=2:continue
                        def ar():qr.qr_all_reduce(ptr,x,out,level,False);return out
                        def check_ar(y):
                            assert torch.isfinite(y).all()
                            err=y.float()-ref
                            if level==0:close(y,ref)
                            else:
                                a,c=limits[level]
                                assert float(err.square().mean().sqrt()/ref.square().mean().sqrt())<=a
                                assert float(err.abs().max()/ref.abs().max())<=c
                        b.measure('hip_qr_'+name,x.shape,dtype,ar,check_ar,cast_bf16_to_fp16=False)
                        quant_ref=ar().clone().float()+res.float()
                        norm_ref=quant_ref*torch.rsqrt(quant_ref.square().mean(-1,keepdim=True)+1e-6)
                        def rms():
                            qr.qr_all_reduce_rmsnorm(ptr,x,res,rout,norm,weight,1e-6,h,level,False)
                            return norm,rout
                        def check_rms(y):close(y[0],norm_ref);close(y[1],quant_ref)
                        b.measure('hip_qr_'+name+'_rms',x.shape,dtype,rms,check_rms,cast_bf16_to_fp16=False)
        finally:
            torch.cuda.synchronize();dist.barrier();qr.qr_destroy(ptr)
    finally:
        torch.cuda.synchronize();dist.barrier();ca.close()


def iris_suite(b):
    from aiter.ops.triton.comms import IrisCommContext,all_gather,reduce_scatter,reduce_scatter_rmsnorm_quant_all_gather
    for m,n in ((4,2048),(128,8192)):
        with IrisCommContext(512<<20) as ctx:
            x=ctx.iris_ctx.empty((m,n),dtype=torch.bfloat16)
            base=(torch.arange(m*n,device='cuda').reshape(m,n).float()%17-8)/16
            inputs=[(base+r/8).bfloat16() for r in range(b.world)]
            x.copy_(inputs[b.rank]);summed=sum(t.float() for t in inputs).bfloat16()
            ref=torch.cat(inputs,0)
            b.measure('iris_ag',x.shape,x.dtype,lambda:all_gather(x,ctx),lambda y:close(y,ref,0),eager=True,output_allocation='included')
            shard=summed.chunk(b.world,0)[b.rank]
            b.measure('iris_rs',x.shape,x.dtype,lambda:reduce_scatter(x,ctx),lambda y:close(y,shard),eager=True,output_allocation='included')
            gamma=torch.ones(n,dtype=x.dtype,device='cuda')
            norm=shard.float()*torch.rsqrt(shard.float().square().mean(-1,keepdim=True)+1e-6)
            full_norm=summed.float()*torch.rsqrt(summed.float().square().mean(-1,keepdim=True)+1e-6)
            rs=ctx.iris_ctx.empty(tuple(shard.shape),dtype=x.dtype)
            nb=torch.empty_like(shard,dtype=torch.float32)
            qb=ctx.iris_ctx.empty(tuple(shard.shape),dtype=torch.float8_e4m3fn)
            for quant in ('none','fp8_per_token'):
                for gather in (False,True):
                    odtype=x.dtype if quant=='none' else torch.float8_e4m3fn
                    go=ctx.iris_ctx.empty((m,n),dtype=odtype) if gather else None
                    def call():return reduce_scatter_rmsnorm_quant_all_gather(x,gamma,ctx=ctx,quant_mode=quant,do_allgather=gather,rs_buffer=rs,norm_buffer=nb,fp8_out=qb,gather_out=go)
                    def check(y):
                        if quant=='none':
                            close(y[0],norm,1e-4)
                            if gather:close(y[2],full_norm)
                        else:
                            scale=(norm.abs().amax(-1,keepdim=True)/448).clamp_min(1e-8)
                            torch.testing.assert_close(y[1].float()*scale,norm,rtol=.1,atol=.03)
                            if gather:
                                scale=(full_norm.abs().amax(-1,keepdim=True)/448).clamp_min(1e-8)
                                torch.testing.assert_close(y[2].float()*scale,full_norm,rtol=.1,atol=.03)
                    b.measure('iris_rms_'+quant+('_ag' if gather else ''),x.shape,x.dtype,call,check,eager=True,output_allocation='preallocated')


def groups_suite(b,layout):
    from aiter.dist import communication_op as comm,parallel_state as ps
    prefixes=['data_parallel','expert_parallel'] if layout=='dp' else ['pipeline_model_parallel'] if layout=='pp' else ['tensor_model_parallel','custom']
    for m,h in ((4,2048),(128,8192)):
        x=torch.full((m,h),b.rank+1.,dtype=torch.bfloat16,device='cuda')
        total=b.world*(b.world+1)/2
        for prefix in prefixes:
            extra={'group':'all'} if prefix=='custom' else {}
            for kind in ('all_reduce','all_gather','reduce_scatter'):
                if prefix=='pipeline_model_parallel' and kind=='reduce_scatter':continue
                fn=getattr(comm,prefix+'_'+kind)
                def call():
                    if kind=='all_reduce':return fn(x.clone(),**extra)
                    return fn(x,use_custom=False,dim=-1,**extra)
                expected=torch.full_like(x,total) if kind=='all_reduce' else torch.cat([torch.full_like(x,r+1.) for r in range(b.world)],-1) if kind=='all_gather' else torch.full((m,h//b.world),total,dtype=x.dtype,device='cuda')
                group=ps.get_custom_group('all') if prefix=='custom' else getattr(ps,'get_'+{'data_parallel':'dp','expert_parallel':'ep','pipeline_model_parallel':'pp','tensor_model_parallel':'tp'}[prefix]+'_group')()
                assert group.world_size==b.world
                b.measure(prefix+'_'+kind,x.shape,x.dtype,call,lambda y:close(y,expected,0),eager=True,group_layout=layout,input_clone=kind=='all_reduce',use_custom=kind=='all_reduce')
            if prefix in ('data_parallel','expert_parallel','pipeline_model_parallel'):
                def broadcast():return getattr(comm,prefix+'_broadcast')(x.clone(),src=0)
                b.measure(prefix+'_broadcast',x.shape,x.dtype,broadcast,lambda y:close(y,torch.ones_like(x),0),eager=True,group_layout=layout,input_clone=True)
                if prefix!='pipeline_model_parallel':
                    def gather():return getattr(comm,prefix+'_gather')(x,dst=0,dim=-1)
                    expected=torch.cat([torch.full_like(x,r+1.) for r in range(b.world)],-1)
                    def check_gather(y):
                        if b.rank==0:close(y,expected,0)
                        else:assert y is None
                    b.measure(prefix+'_gather',x.shape,x.dtype,gather,check_gather,eager=True,group_layout=layout)
        if layout=='pp' and b.world==2:
            def pair():
                if b.rank==0:comm.pipeline_model_parallel_send(x,dst=1);return x
                return comm.pipeline_model_parallel_recv(x.shape,x.dtype,src=0)
            b.measure('pipeline_send_recv',x.shape,x.dtype,pair,lambda y:close(y,torch.ones_like(x),0),eager=True,group_layout=layout)


def provenance(args):
    import aiter
    prop=torch.cuda.get_device_properties(torch.cuda.current_device())
    modules={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and ('module_custom_all_reduce' in name or 'module_quick_all_reduce' in name or 'module_fused_ar_mhc' in name):
            p=Path(file)
            if p.is_file():modules[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    report=dict(aiter=aiter.__file__,torch=torch.__version__,hip=torch.version.hip,
                device=prop.name,arch=prop.gcnArchName,cu=prop.multi_processor_count,
                visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'),
                native_modules=modules,arguments=vars(args),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for pkg in ('triton','iris'):
        try:report[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:report[pkg]='not installed as distribution'
    if 'iris' in sys.modules:report['iris_source']=str(Path(sys.modules['iris'].__file__).resolve())
    Path(args.output,f'provenance_rank{dist.get_rank()}.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',choices=['car','hip','iris','groups_tp','groups_dp','groups_pp'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=30)
    p.add_argument('--graph-ops',type=int,default=16)
    args=p.parse_args()
    if args.samples<2 or args.graph_ops<1:p.error('samples>=2 and graph-ops>=1 required')
    local,rank,world=(int(os.environ[k]) for k in ('LOCAL_RANK','RANK','WORLD_SIZE'))
    if world!=2:p.error('this representative benchmark currently requires TP2')
    torch.cuda.set_device(local)
    grouped=args.suite.startswith('groups_')
    if grouped:
        from aiter.ops.communication import init_dist_env
        layout=args.suite.removeprefix('groups_')
        init_dist_env(world_size=world if layout=='tp' else 1,
                      rankID=rank if layout!='dp' else 0,local_rank=local,backend='nccl',
                      data_parallel_size=world if layout=='dp' else 1,
                      data_parallel_rank=rank if layout=='dp' else 0,
                      pipeline_model_parallel_size=world if layout=='pp' else 1,
                      custom_group_config={'all':list(range(world))} if layout=='tp' else None)
    else:dist.init_process_group('gloo',timeout=timedelta(seconds=180))
    bench=Bench(args)
    try:
        if args.suite=='car':car_ar_suite(bench)
        elif args.suite=='hip':car_suite(bench)
        elif args.suite=='iris':iris_suite(bench)
        else:groups_suite(bench,layout)
        provenance(args)
        Path(args.output,f'PASS_rank{rank}').write_text(str(len(bench.rows))+'\n')
    finally:
        if grouped:
            from aiter.ops.communication import destroy_dist_env
            destroy_dist_env()
        else:dist.destroy_process_group()


if __name__=='__main__':main()

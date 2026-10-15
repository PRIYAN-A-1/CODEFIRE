"""Measured single training-step probe. Use a fresh process before full training."""
import argparse,gc,json,time
import torch,yaml
from src.model.transformer import ModelConfig,Transformer
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='configs/model_23m.yaml');a=p.parse_args()
    if not torch.cuda.is_available():print(json.dumps({'status':'unavailable','reason':'CUDA not available; batch size not measured'}));raise SystemExit(0)
    c=ModelConfig(**yaml.safe_load(open(a.model)));rows=[]
    for seq in (128,256,512):
        if seq>c.max_context_length:continue
        for batch in (1,2,4):
            m=opt=None
            try:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();m=Transformer(c).cuda();m.gradient_checkpointing=True;opt=torch.optim.AdamW(m.parameters());x=torch.randint(c.vocab_size,(batch,seq),device='cuda');tic=time.time()
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                scaler=torch.amp.GradScaler('cuda',enabled=dtype==torch.float16)
                with torch.autocast('cuda',dtype=dtype):loss=m(x,x)[1]
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize()
                peak=torch.cuda.max_memory_allocated();total=torch.cuda.get_device_properties(0).total_memory
                rows.append({'batch':batch,'sequence':seq,'peak_bytes':peak,'headroom_ok':peak<total*.8,'seconds':time.time()-tic})
            except torch.cuda.OutOfMemoryError:rows.append({'batch':batch,'sequence':seq,'status':'oom'});break
            finally:del m,opt;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(rows,indent=2))

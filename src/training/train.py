import argparse,contextlib,hashlib,json,math,random,signal,time
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch,yaml
from src.model.transformer import ModelConfig,Transformer
from src.data.dataset import TokenDataset
from src.training.checkpoint import save,load,rng_state,restore_rng

def train(c,resume=None):
    torch.set_num_threads(c.get('cpu_threads',2));seed=c.get('seed',42);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() and c.get('device','auto')!='cpu' else 'cpu'
    mc=ModelConfig(**yaml.safe_load(open(c['model'])));model=Transformer(mc).to(device);model.gradient_checkpointing=c.get('gradient_checkpointing',False)
    ds=TokenDataset(c['dataset'],'train',c['sequence_length']);val=TokenDataset(c['dataset'],'validation',c['sequence_length'])
    checksum=hashlib.sha256(Path(c['tokenizer']).read_bytes()).hexdigest()
    if ds.manifest['tokenizer_sha256']!=checksum:raise ValueError('Dataset tokenizer mismatch')
    from tokenizers import Tokenizer
    if Tokenizer.from_file(c['tokenizer']).get_vocab_size()>mc.vocab_size:raise ValueError('Tokenizer exceeds model vocabulary')
    manifest_hash=hashlib.sha256((Path(c['dataset'])/'manifest.json').read_bytes()).hexdigest()
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),weight_decay=.1)
    precision=c.get('mixed_precision','auto')
    if precision=='auto':precision=('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device=='cuda' else 'fp32'
    if device=='cpu' and precision!='fp32':raise ValueError('Use fp32 for CPU smoke test')
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[precision]
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and precision=='fp16')
    gen=torch.Generator().manual_seed(seed);vg=torch.Generator().manual_seed(seed+1);start=0;tokens=0
    if resume:
        s=load(resume)
        if s.get('stage')=='sft':raise ValueError('SFT checkpoint cannot resume the pretraining optimizer; use a new training experiment')
        if s['model_config']!=asdict(mc) or s['tokenizer_sha256']!=checksum or s['manifest_sha256']!=manifest_hash:raise ValueError('Resume model/tokenizer/dataset mismatch')
        for key in ['sequence_length','micro_batch_size','gradient_accumulation_steps','learning_rate','steps']:
            if s['training_config'][key]!=c[key]:raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler']);start=s['step'];tokens=s['tokens'];gen.set_state(s['sampler']);vg.set_state(s['val_sampler']);restore_rng(s['rng'])
    root=Path(c['checkpoint_dir'])
    if (root/'latest').exists() and not resume:raise ValueError('Existing experiment. Resume explicitly or choose a new checkpoint_dir.')
    root.mkdir(parents=True,exist_ok=True);log=Path(c.get('log','logs/train.jsonl'));log.parent.mkdir(parents=True,exist_ok=True)
    stop=[False]
    def halt(*_):stop[0]=True
    old={}
    for sig in (signal.SIGINT,signal.SIGTERM):old[sig]=signal.signal(sig,halt)
    step=start
    def persist():
        save(root/f'step-{step:08}.pt',{'format_version':1,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'model_config':asdict(mc),'training_config':c,'step':step,'tokens':tokens,'sampler':gen.get_state(),'val_sampler':vg.get_state(),'rng':rng_state(),'tokenizer_sha256':checksum,'manifest_sha256':manifest_hash,'scheduler':{'step':step,'total_steps':c['steps']}})
        files=sorted(root.glob('step-*.pt'))
        for f in files[:-c.get('checkpoint_keep_last',2)]:f.unlink();f.with_suffix('.sha256').unlink(missing_ok=True)
    try:
        for step in range(start+1,c['steps']+1):
            tic=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;warm=max(1,int(c['steps']*c.get('warmup_ratio',.02)))
            factor=step/warm if step<=warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/max(1,c['steps']-warm)))
            for group in opt.param_groups:group['lr']=c['learning_rate']*factor
            for _ in range(c['gradient_accumulation_steps']):
                x,y=ds.batch(c['micro_batch_size'],gen,device)
                with torch.autocast(device_type=device,dtype=dtype,enabled=precision!='fp32'):
                    _,loss,_=model(x,y);scaled=loss/c['gradient_accumulation_steps']
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
                scaler.scale(scaled).backward();loss_sum+=loss.item()/c['gradient_accumulation_steps']
            scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update()
            nt=c['micro_batch_size']*c['sequence_length']*c['gradient_accumulation_steps'];tokens+=nt
            row={'step':step,'tokens':tokens,'loss':loss_sum,'lr':opt.param_groups[0]['lr'],'tokens_per_second':nt/(time.perf_counter()-tic),'validation_loss':None,'gpu_peak_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None}
            if step%c.get('eval_every_steps',50)==0 or step==c['steps']:
                model.eval()
                with torch.no_grad():
                    x,y=val.batch(c['micro_batch_size'],vg,device);row['validation_loss']=model(x,y)[1].item()
            with log.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if step%c.get('checkpoint_every_steps',100)==0 or step==c['steps'] or stop[0]:persist()
            if stop[0]:break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA OOM: last complete checkpoint remains available. Reduce microbatch/context in a new experiment.') from exc
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    return {'step':step,'tokens':tokens,'parameters':model.parameter_count()}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/train_23m.yaml');p.add_argument('--resume');a=p.parse_args();print(train(yaml.safe_load(open(a.config)),a.resume))

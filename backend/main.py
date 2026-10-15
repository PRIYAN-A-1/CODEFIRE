import asyncio,hashlib,json,os,subprocess,sys,threading
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
import yaml
import torch
torch.set_num_threads(int(os.environ.get("PRIYAN_CPU_THREADS","2")))
from src.system.hardware import report
from src.inference.generate import Engine
ROOT=Path(__file__).resolve().parents[1]
ALLOWED={name:ROOT/f'configs/train_{name}.yaml' for name in ('smoke','23m','134m')}
lock=threading.Lock();job=None;job_log=None;active_preset='smoke'

def local(path):return ROOT/path
def checkpoint_path():return Path(os.environ.get('PRIYAN_CHECKPOINT',str(local('checkpoints/latest'))))
def tokenizer_path():return Path(os.environ.get('PRIYAN_TOKENIZER',str(local('artifacts/tokenizer.json'))))

def jsonfile(path,default):
    p=local(path);return json.loads(p.read_text()) if p.exists() else default

app=FastAPI(title='PRIYAN CODE AI',version='0.1.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type','Authorization'])
@app.middleware('http')
async def protect(request:Request,call_next):
    key=os.environ.get('PRIYAN_API_KEY')
    origin=request.headers.get('origin')
    if origin and origin not in ('http://localhost:5173','http://127.0.0.1:5173'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Origin not allowed'},status_code=403)
    host=request.headers.get('host','').split(':')[0]
    if host not in ('127.0.0.1','localhost','testserver'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Loopback service only; use an authenticated reverse proxy with loopback Host'},status_code=403)
    if key and request.headers.get('authorization')!='Bearer '+key:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Unauthorized'},status_code=401)
    length=request.headers.get('content-length')
    if length and int(length)>65536:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Request too large'},status_code=413)
    return await call_next(request)

def active():return job is not None and job.poll() is None
class Generate(BaseModel):
    prompt:str=Field(min_length=1,max_length=12000)
    max_new_tokens:int=Field(default=64,ge=1,le=512)
    temperature:float=Field(default=.7,ge=0,le=2)
    top_k:int=Field(default=40,ge=0,le=1000)
    top_p:float=Field(default=.95,gt=0,le=1)
    seed:int=42
class Start(BaseModel):
    preset:str='smoke'
    resume:bool=False

def engine():
    if active():raise HTTPException(409,'Training owns model resources')
    if not checkpoint_path().exists():raise HTTPException(503,'Model not trained/loaded')
    try:return Engine(checkpoint_path(),tokenizer_path())
    except (OSError,ValueError,RuntimeError) as exc:raise HTTPException(503,str(exc))
@app.get('/health')
def health():return {'status':'ok','model_available':checkpoint_path().exists(),'training':active()}
@app.get('/api/system/hardware')
def hardware():return report(ROOT)
@app.get('/api/system/storage')
def storage():return {'paths':{s:str(local(s)) for s in ['data/tokenized','checkpoints','artifacts']},'disk_free_bytes':report(ROOT)['disk_free_bytes'],'artificial_budget':None}
@app.get('/api/models')
def models():
    result=[]
    for name in ('smoke','23m','134m'):
        c=yaml.safe_load(local(f'configs/model_{name}.yaml').read_text());d=c['hidden_size'];n=c['num_layers'];v=c['vocab_size'];f=c['intermediate_size'];count=v*d*(1 if c['tie_word_embeddings'] else 2)+n*(4*d*d+3*d*f+2*d)+d
        result.append({'id':name,'config':c,'parameters':count})
    return result
@app.get('/api/models/{model_id}')
def model_info(model_id:str):
    for m in models():
        if m['id']==model_id:return m
    raise HTTPException(404,'Unknown model')
@app.get('/api/datasets')
def datasets():return {'manifest':jsonfile('data/tokenized/manifest.json',None),'quality':jsonfile('data/clean/collection_report.json',None),'config':yaml.safe_load(local('configs/dataset.yaml').read_text())}
@app.post('/api/datasets/validate')
def validate_dataset():
    from src.data.dataset import TokenDataset
    try:
        ds=TokenDataset(local('data/tokenized'),'train',32)
        return {'valid':True,'train_sequences':ds.length}
    except (OSError,ValueError) as exc:raise HTTPException(400,str(exc))
@app.get('/api/training/status')
def status():
    p=local('logs/train.jsonl' if active_preset=='smoke' else f'logs/{active_preset}/train.jsonl');rows=[]
    if p.exists():
        for line in p.read_text().splitlines()[-100:]:
            try:rows.append(json.loads(line))
            except ValueError:pass
    return {'running':active(),'pid':job.pid if active() else None,'returncode':job.poll() if job else None,'metrics':rows,'latest':rows[-1] if rows else None}
@app.post('/api/training/start')
def start(a:Start):
    global job,job_log,active_preset
    if a.preset not in ALLOWED:raise HTTPException(400,'Unknown preset')
    if not lock.acquire(False):raise HTTPException(409,'Model resources busy')
    try:
        if active():raise HTTPException(409,'Training already running')
        cfg=yaml.safe_load(ALLOWED[a.preset].read_text());latest=local(cfg['checkpoint_dir'])/'latest'
        if not local(cfg['dataset']).joinpath('manifest.json').exists():raise HTTPException(400,'Prepare the matching dataset and tokenizer before training this preset')
        if latest.exists() and not a.resume:raise HTTPException(409,'Existing experiment: resume or configure a new checkpoint directory through CLI')
        cmd=[sys.executable,'-m','src.training.train','--config',str(ALLOWED[a.preset])]
        if a.resume:
            if not latest.exists():raise HTTPException(400,'No checkpoint to resume')
            cmd+=['--resume',str(latest)]
        local('logs').mkdir(exist_ok=True)
        if job_log:job_log.close()
        active_preset=a.preset;job_log=open(local('logs/job.log'),'a');job=subprocess.Popen(cmd,cwd=ROOT,stdout=job_log,stderr=subprocess.STDOUT)
        return {'pid':job.pid,'status':'started; monitor returncode and logs'}
    finally:lock.release()
@app.post('/api/training/resume')
def resume(a:Start):a.resume=True;return start(a)
@app.post('/api/training/stop')
def stop():
    if active():job.terminate();return {'status':'safe stop requested; checkpoint after complete step'}
    return {'status':'idle'}
@app.get('/api/checkpoints')
def checkpoints():return [{'file':p.name,'bytes':p.stat().st_size} for p in sorted(local('checkpoints').rglob('*.pt'))]
@app.post('/api/generate')
def generate(a:Generate):
    if not lock.acquire(False):raise HTTPException(409,'Model resources busy')
    try:
        e=engine()
        try:return {'text':e.generate(**a.model_dump()),'model':e.model.config.name,'step':e.state['step'],'stage':e.state.get('stage','base code continuation')}
        except ValueError as exc:raise HTTPException(400,str(exc))
    finally:lock.release()
@app.post('/api/generate/stream')
def stream(a:Generate):
    if not lock.acquire(False):raise HTTPException(409,'Model resources busy')
    try:
        e=engine()
        if len(e.tokenizer.encode(a.prompt).ids)+a.max_new_tokens>e.model.config.max_context_length:raise HTTPException(400,'Prompt plus output exceeds context')
    except Exception:lock.release();raise
    def events():
        try:
            for text in e.stream(**a.model_dump()):yield 'data: '+json.dumps({'text':text})+'\n\n'
            yield 'data: '+json.dumps({'done':True})+'\n\n'
        finally:lock.release()
    return StreamingResponse(events(),media_type='text/event-stream')
@app.get('/api/evaluation/results')
def results():return jsonfile('evaluations/latest.json',{'status':'not_evaluated'})
@app.post('/api/evaluation/run')
def evaluate():
    global job,job_log
    if not lock.acquire(False):raise HTTPException(409,'Model resources busy')
    try:
        if active():raise HTTPException(409,'Training/evaluation busy')
        if not checkpoint_path().exists():raise HTTPException(503,'No checkpoint')
        local('logs').mkdir(exist_ok=True)
        if job_log:job_log.close()
        job_log=open(local('logs/evaluation-job.log'),'a');job=subprocess.Popen([sys.executable,'-m','src.evaluation.evaluate','--checkpoint',str(checkpoint_path()),'--tokenizer',str(tokenizer_path()),'--dataset',os.environ.get('PRIYAN_DATASET','data/tokenized')],cwd=ROOT,stdout=job_log,stderr=subprocess.STDOUT)
        return {'status':'evaluation_started','pid':job.pid}
    finally:lock.release()

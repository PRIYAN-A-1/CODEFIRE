import hashlib,json,os,random
from pathlib import Path
import numpy as np,torch
from src.system.storage import require_space

def save(path,state):
    path=Path(path);require_space(path.parent,sum(v.numel()*v.element_size() for v in state['model'].values())*4)
    temp=path.with_suffix('.tmp');torch.save(state,temp)
    with open(temp,'rb') as f:os.fsync(f.fileno())
    sha=hashlib.sha256(temp.read_bytes()).hexdigest();temp.replace(path)
    hp=path.with_suffix('.sha256');ht=hp.with_suffix('.tmp');ht.write_text(sha);ht.replace(hp)
    latest=path.parent/'latest';tmp=latest.with_suffix('.tmp');tmp.write_text(path.name);tmp.replace(latest)

def load(path):
    path=Path(path)
    if path.name=='latest':path=path.parent/path.read_text().strip()
    if not path.is_file():raise FileNotFoundError(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Checkpoint checksum mismatch')
    # Only load trusted local checkpoints: optimizer and RNG state require pickle.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format_version')!=1:raise ValueError('Unsupported checkpoint format')
    return state

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])

import pytest,torch
from src.inference.generate import Engine
from src.agent.repository import resolve,propose
from src.training.checkpoint import save

def test_seeded_generation_and_stop(model,corpus,tmp_path):
    import hashlib
    from dataclasses import asdict
    from pathlib import Path
    p=tmp_path/'a.pt';save(p,{'format_version':1,'model':model.state_dict(),'model_config':asdict(model.config),'tokenizer_sha256':hashlib.sha256(Path(corpus['tokenizer']).read_bytes()).hexdigest()})
    e=Engine(p,corpus['tokenizer']);a=e.generate('def',max_new_tokens=8,seed=10);b=e.generate('def',max_new_tokens=8,seed=10);assert a==b
    with pytest.raises(ValueError):e.generate('def',max_new_tokens=129)

def test_traversal_blocked(tmp_path):
    with pytest.raises(ValueError):resolve(tmp_path,'../secret.txt')
def test_diff_no_mutation(tmp_path):
    (tmp_path/'a.py').write_text('old\n');patch=propose(tmp_path,'a.py','new\n');assert '-old' in patch and '+new' in patch;assert (tmp_path/'a.py').read_text()=='old\n'
def test_sandbox_no_host_fallback(monkeypatch):
    import src.agent.sandbox as sandbox
    monkeypatch.setattr(sandbox.shutil,'which',lambda x:None)
    with pytest.raises(RuntimeError,match='Docker'):sandbox.test_python('x=1','assert x==1')

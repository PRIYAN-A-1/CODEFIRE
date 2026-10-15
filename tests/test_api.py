import pytest
from fastapi.testclient import TestClient
from backend import main
@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setattr(main,'ROOT',tmp_path);monkeypatch.delenv('PRIYAN_API_KEY',raising=False)
    return TestClient(main.app)
def test_health(client):assert client.get('/health').json()['status']=='ok'
def test_untrained_model(client):assert client.post('/api/generate',json={'prompt':'hello'}).status_code==503
def test_invalid_payload(client):assert client.post('/api/generate',json={'prompt':'','max_new_tokens':-3}).status_code==422
def test_authentication(client,monkeypatch):
    monkeypatch.setenv('PRIYAN_API_KEY','local-test-key');assert client.get('/health').status_code==401;assert client.get('/health',headers={'Authorization':'Bearer local-test-key'}).status_code==200
def test_origin_guard(client):assert client.post('/api/training/start',json={},headers={'Origin':'https://untrusted.example'}).status_code==403
def test_host_guard(client):assert client.get('/health',headers={'host':'untrusted.example'}).status_code==403
def test_unknown_preset(client):assert client.post('/api/training/start',json={'preset':'../../etc'}).status_code==400
def test_oversized(client):assert client.post('/api/generate',content='x'*70000).status_code==413
def test_status_empty(client):assert client.get('/api/training/status').json()['latest'] is None


def test_real_checkpoint_generation_and_stream(client,model,corpus,tmp_path):
    import hashlib
    from dataclasses import asdict
    from pathlib import Path
    from src.training.checkpoint import save
    import shutil
    artifacts=tmp_path/'artifacts';artifacts.mkdir(exist_ok=True);shutil.copy(corpus['tokenizer'],artifacts/'tokenizer.json')
    state={'format_version':1,'model':model.state_dict(),'model_config':asdict(model.config),'tokenizer_sha256':hashlib.sha256(Path(corpus['tokenizer']).read_bytes()).hexdigest(),'step':1}
    save(tmp_path/'checkpoints/step-00000001.pt',state)
    result=client.post('/api/generate',json={'prompt':'def f(x):','max_new_tokens':5,'temperature':0})
    assert result.status_code==200 and isinstance(result.json()['text'],str)
    result=client.post('/api/generate/stream',json={'prompt':'def f(x):','max_new_tokens':5,'temperature':0})
    assert result.status_code==200 and '"done": true' in result.text

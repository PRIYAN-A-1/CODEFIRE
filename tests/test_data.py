import json,pytest,torch
from pathlib import Path
from tokenizers import Tokenizer
from src.data.collect import collect,SECRET
from src.data.prepare import prepare
from src.data.dataset import TokenDataset
@pytest.mark.parametrize('text',['def f(x):\n\treturn x + 1\n','வணக்கம் 🌏 café\n','  foo\r\n    bar\n','rare_identifier_374843 != 1;'])
def test_tokenizer_roundtrip(corpus,text):
    tok=Tokenizer.from_file(corpus['tokenizer']);assert tok.decode(tok.encode(text).ids)==text

def test_explicit_permission_required(tmp_path):
    with pytest.raises(ValueError):collect({'records':str(tmp_path/'x'),'sources':[{'approved':False}]})
def test_secret_detection():assert SECRET.search('api_key = "private-secret-12345"')
def test_shards_and_reuse(corpus):
    ds=TokenDataset(corpus['output'],'train',8);x,y=ds.batch(2,torch.Generator().manual_seed(1),'cpu');assert x.shape==(2,8);assert torch.equal(x[:,1:],y[:,:-1]);assert prepare(corpus)['records']==3

def test_corrupted_shard(corpus):
    m=json.loads((Path(corpus['output'])/'manifest.json').read_text());p=Path(corpus['output'])/next(s['file'] for s in m['shards'] if s['split']=='train');p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):TokenDataset(corpus['output'],'train',8)
def test_missing_split(corpus):
    with pytest.raises(ValueError):TokenDataset(corpus['output'],'validation',10000)
def test_dedup_and_secret_filters(tmp_path):
    p=tmp_path/'repo';p.mkdir();text='def answer():\n    return 101\n';(p/'a.py').write_text(text);(p/'b.py').write_text(text);(p/'c.py').write_text('api_key = "sensitive-value-12345"')
    c={'records':str(tmp_path/'records.jsonl'),'sources':[{'path':str(p),'repository':'x','approved':True,'license_spdx':'MIT'}]};assert collect(c)==1

def test_repo_splits_isolated(corpus):
    records=[json.loads(s) for s in Path(corpus['records']).read_text().splitlines()];splits={}
    for r in records:splits.setdefault(r['repository'],set()).add(r['split'])
    assert all(len(s)==1 for s in splits.values())


def test_frozen_tokenizer(corpus):
    from src.tokenizer.train_tokenizer import train
    train(corpus['records'],corpus['tokenizer'],512)
    with pytest.raises(ValueError,match='frozen'):train(corpus['records'],corpus['tokenizer'],1024)

def test_conflicting_repo_split_rejected(tmp_path):
    path=tmp_path/'repo';path.mkdir();(path/'a.py').write_text('def identity(value):\n    return value\n')
    source={'path':str(path),'repository':'same','license_spdx':'MIT','approved':True}
    with pytest.raises(ValueError,match='repository'):collect({'records':str(tmp_path/'records'),'sources':[dict(source,split='train'),dict(source,split='test')]})

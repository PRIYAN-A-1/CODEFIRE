import json,pytest,torch,yaml
from pathlib import Path
from src.training.train import train
from src.training.checkpoint import load,save
from src.training.sft import format_example
from tokenizers import Tokenizer

def configuration(corpus,tmp_path):
    mc=tmp_path/'model.yaml';mc.write_text(json.dumps(dict(vocab_size=512,hidden_size=32,num_layers=1,num_attention_heads=4,intermediate_size=64,max_context_length=128)))
    return dict(model=str(mc),dataset=corpus['output'],tokenizer=corpus['tokenizer'],sequence_length=8,micro_batch_size=1,gradient_accumulation_steps=2,learning_rate=.001,steps=4,checkpoint_dir=str(tmp_path/'ckpt'),log=str(tmp_path/'log.jsonl'),checkpoint_every_steps=2,eval_every_steps=2,checkpoint_keep_last=5,device='cpu')
def test_exact_resume(corpus,tmp_path):
    c=configuration(corpus,tmp_path);train(c);final=load(Path(c['checkpoint_dir'])/'latest');checkpoint=Path(c['checkpoint_dir'])/'step-00000002.pt';train(c,str(checkpoint));resumed=load(Path(c['checkpoint_dir'])/'latest')
    for k in final['model']:assert torch.equal(final['model'][k],resumed['model'][k])
    assert final['tokens']==resumed['tokens']==64

def test_corrupt_checkpoint(tmp_path):
    p=tmp_path/'bad.pt';p.write_bytes(b'bad');p.with_suffix('.sha256').write_text('wrong')
    with pytest.raises(ValueError):load(p)
def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):load(tmp_path/'absent.pt')
def test_resume_config_guard(corpus,tmp_path):
    c=configuration(corpus,tmp_path);train(c);c['sequence_length']=16
    with pytest.raises(ValueError,match='configuration'):train(c,str(Path(c['checkpoint_dir'])/'latest'))
def test_sft_masks_prompt(corpus):
    tok=Tokenizer.from_file(corpus['tokenizer']);x,y=format_example(tok,{'instruction':'Write sum','response':'return 3'},128);assert -100 in y;assert y[-1]==tok.token_to_id('<|eos|>');assert len(x)==len(y)

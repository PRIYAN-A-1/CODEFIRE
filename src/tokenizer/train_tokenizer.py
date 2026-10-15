import argparse, hashlib, json
from pathlib import Path
import yaml
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
SPECIAL=['<unk>','<|bos|>','<|eos|>','<|pad|>','<|fim_prefix|>','<|fim_middle|>','<|fim_suffix|>','<|system|>','<|user|>','<|assistant|>','<|end|>']
def train(records,output,vocab=32000):
    p=Path(output); report=p.with_name('tokenizer_report.json')
    source_hash=hashlib.sha256(Path(records).read_bytes()).hexdigest()
    if p.exists():
        old=json.loads(report.read_text())
        if old.get('records_sha256')!=source_hash or old.get('requested_vocab')!=vocab:raise ValueError('Tokenizer is frozen: use a new artifact path for changed corpus/vocabulary')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=old['sha256']:raise ValueError('Tokenizer integrity mismatch')
        return Tokenizer.from_file(str(p))
    t=Tokenizer(models.BPE(unk_token='<unk>'))
    t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    t.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=vocab,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t.train_from_iterator((record['content'] for line in open(records,encoding='utf-8') if (record:=json.loads(line)).get('split','train')=='train'),trainer)
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);t.save(str(p))
    info={'records_sha256':source_hash,'requested_vocab':vocab,'version':1,'vocab_size':t.get_vocab_size(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':str(records),'special_tokens':SPECIAL}
    p.with_name('tokenizer_report.json').write_text(json.dumps(info,indent=2))
    p.with_name('tokenizer_config.json').write_text(json.dumps({'add_bos_token':False,'add_eos_token':False,'model_max_length':None,'special_tokens':SPECIAL},indent=2))
    return t
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/dataset.yaml');a=p.parse_args();c=yaml.safe_load(open(a.config));train(c['records'],c['tokenizer'],c.get('vocab_size',32000))
